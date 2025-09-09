# src/train_gpu.py
#!/usr/bin/env python3
"""
GPU最適化MIDI RAG学習スクリプト
CUDA 12.x対応完全版
"""

import os
import sys
import yaml
import argparse
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Any, Optional
import time
import psutil
import GPUtil

# カスタムモジュール
sys.path.append('src')
from models.gpu_optimized_model import GPUOptimizedMIDIRAG, MIDIComposerRAGConfig, GPUProfiler
from utils.gpu_dataloader import MIDIDatasetGPU, GPUCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUTrainer:
    """GPU最適化トレーナー"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.config.get('training', {}).get('fp16', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # GPU情報表示
        self._display_gpu_info()
        
        # モデル、オプティマイザー等の初期化
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # 学習統計
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # 監視設定
        self.use_wandb = self.config.get('monitoring', {}).get('wandb', {}).get('enabled', False)
        
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _display_gpu_info(self):
        """GPU情報表示"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")
            logger.info(f"Mixed Precision: {self.use_amp}")
        else:
            logger.warning("CUDA not available, using CPU")
    
    def setup_model(self) -> GPUOptimizedMIDIRAG:
        """モデルセットアップ"""
        # モデル設定
        model_config = MIDIComposerRAGConfig(
            vocab_size=self.config.get('model', {}).get('vocab_size', 512),
            hidden_size=self.config.get('model', {}).get('hidden_size', 768),
            num_attention_heads=self.config.get('model', {}).get('num_attention_heads', 12),
            num_hidden_layers=self.config.get('model', {}).get('num_hidden_layers', 12),
            max_position_embeddings=self.config.get('model', {}).get('max_position_embeddings', 2048),
            dropout=self.config.get('model', {}).get('dropout', 0.1),
            use_flash_attention=self.config.get('model', {}).get('use_flash_attention', True)
        )
        
        # モデル作成
        self.model = GPUOptimizedMIDIRAG(model_config).to(self.device)
        
        # 複数GPU対応
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # PyTorch Compile (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.config.get('gpu', {}).get('compile_model', True):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with PyTorch 2.0")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # 混合精度対応
        if self.use_amp:
            self.model = self.model.half()
        
        # パラメータ数表示
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def setup_optimizer(self):
        """オプティマイザーセットアップ"""
        training_config = self.config.get('training', {})
        
        # AdamW オプティマイザー
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 2e-5),
            betas=(training_config.get('adam_beta1', 0.9), training_config.get('adam_beta2', 0.999)),
            eps=training_config.get('adam_epsilon', 1e-8),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        return self.optimizer
    
    def setup_scheduler(self, num_training_steps: int):
        """学習率スケジューラーセットアップ"""
        training_config = self.config.get('training', {})
        warmup_steps = int(num_training_steps * training_config.get('warmup_ratio', 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return self.scheduler
    
    def setup_dataloaders(self):
        """データローダーセットアップ"""
        training_config = self.config.get('training', {})
        
        # データセット作成
        train_dataset = MIDIDatasetGPU(
            data_dir="data/processed",
            max_sequence_length=self.config.get('model', {}).get('max_position_embeddings', 2048),
            preload_to_memory=True
        )
        
        # コライダー
        collator = GPUCollator(pad_token_id=0)
        
        # データローダー
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_config.get('batch_size', 16),
            shuffle=True,
            num_workers=training_config.get('dataloader_num_workers', 4),
            pin_memory=training_config.get('pin_memory', True),
            collate_fn=collator,
            persistent_workers=True,
            prefetch_factor=training_config.get('prefetch_factor', 2),
            drop_last=True
        )
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Training batches: {len(self.train_dataloader)}")
        
        return self.train_dataloader
    
    def setup_monitoring(self):
        """監視システムセットアップ"""
        if self.use_wandb:
            wandb_config = self.config.get('monitoring', {}).get('wandb', {})
            
            wandb.init(
                project=wandb_config.get('project', 'midi-composer-rag'),
                entity=wandb_config.get('entity', None),
                config=self.config,
                name=f"midi-rag-gpu-{int(time.time())}"
            )
            
            # モデル構造をログ
            wandb.watch(self.model, log="all", log_freq=100)
    
    def train_epoch(self) -> Dict[str, float]:
        """1エポック学習"""
        self.model.train()
        
        training_config = self.config.get('training', {})
        gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        max_grad_norm = training_config.get('max_grad_norm', 1.0)
        
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        # プログレスバー
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for step, batch in enumerate(progress_bar):
            # バッチをGPUに転送
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
                
                # 勾配累積の場合は損失を平均化
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 勾配累積
            if (step + 1) % gradient_accumulation_steps == 0:
                # 勾配クリッピング
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # 統計更新
            total_loss += loss.item() * gradient_accumulation_steps
            avg_loss = total_loss / (step + 1)
            
            # プログレスバー更新
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'step': self.global_step
            })
            
            # ログ出力
            if step % 100 == 0:
                GPUProfiler.log_memory_usage(f"Epoch {self.epoch}, Step {step}")
                
                # Wandb ログ
                if self.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/global_step': self.global_step,
                        'system/gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                        'system/gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                    })
            
            # メモリクリーンアップ（必要に応じて）
            if step % 1000 == 0:
                GPUProfiler.clear_cache()
        
        return {'train_loss': total_loss / num_batches}
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """チェックポイント保存"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 保存するデータ
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 通常のチェックポイント
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # ベストモデル
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイント読み込み"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['metrics'].get('train_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self, num_epochs: int = 10, resume_from: str = None):
        """メイン学習ループ"""
        logger.info("🚀 Starting GPU-optimized training...")
        
        # セットアップ
        self.setup_model()
        self.setup_dataloaders()
        self.setup_optimizer()
        
        # スケジューラーセットアップ
        total_steps = len(self.train_dataloader) * num_epochs
        self.setup_scheduler(total_steps)
        
        # 監視セットアップ
        self.setup_monitoring()
        
        # チェックポイント読み込み
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            logger.info(f"Resumed from epoch {self.epoch}")
        
        # 学習開始
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 1エポック学習
            epoch_start_time = time.time()
            metrics = self.train_epoch()
            epoch_time = time.time() - epoch_start_time
            
            # ログ出力
            train_loss = metrics['train_loss']
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            logger.info(f"Average loss: {train_loss:.4f}")
            
            # ベストモデル判定
            is_best = train_loss < self.best_loss
            if is_best:
                self.best_loss = train_loss
            
            # チェックポイント保存
            self.save_checkpoint(metrics, is_best)
            
            # Wandb ログ
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'train/epoch_time': epoch_time,
                    'train/is_best': is_best
                })
        
        # 学習完了
        total_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {total_time:.2f}s")
        logger.info(f"Best loss: {self.best_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()
    
    def evaluate_model(self):
        """モデル評価（簡易版）"""
        self.model.eval()
        
        with torch.no_grad():
            # ダミー評価（実際の実装では適切な評価データセットが必要）
            sample_input = torch.randint(0, 512, (1, 64), device=self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(input_ids=sample_input)
            
            logger.info(f"Evaluation completed. Output shape: {outputs['logits'].shape}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="GPU-optimized MIDI RAG Training")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory")
    parser.add_argument("--profile", help="GPU optimization profile")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--dry_run", action="store_true", help="Test setup without training")
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # トレーナー初期化
        trainer = GPUTrainer(config_path=args.config)
        
        # GPU プロファイル適用
        if args.profile:
            trainer._apply_gpu_profile(args.profile)
        
        # Wandb 設定
        if args.wandb:
            trainer.use_wandb = True
        
        # Dry run（設定テスト）
        if args.dry_run:
            logger.info("🧪 Running dry run...")
            
            # モデル、データローダーのセットアップテスト
            trainer.setup_model()
            trainer.setup_dataloaders()
            trainer.setup_optimizer()
            trainer.setup_scheduler(100)  # ダミー
            
            # 簡単なforward テスト
            trainer.evaluate_model()
            
            logger.info("✅ Dry run completed successfully!")
            return
        
        # 学習実行
        logger.info("🚀 Starting training...")
        trainer.train(num_epochs=args.epochs, resume_from=args.resume)
        
        logger.info("🎉 Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


# GPU プロファイル設定
def apply_gpu_profile(trainer, profile_name: str):
    """GPU最適化プロファイルを適用"""
    profiles = {
        'enterprise': {  # A100, H100
            'batch_size': 32,
            'gradient_accumulation_steps': 2,
            'fp16': False,
            'bf16': True,
            'gradient_checkpointing': False,
            'dataloader_num_workers': 8
        },
        'high_end': {  # RTX 4090, 3090
            'batch_size': 16,
            'gradient_accumulation_steps': 4,
            'fp16': True,
            'gradient_checkpointing': False,
            'dataloader_num_workers': 6
        },
        'mainstream': {  # RTX 4080, 3080
            'batch_size': 12,
            'gradient_accumulation_steps': 6,
            'fp16': True,
            'gradient_checkpointing': True,
            'dataloader_num_workers': 4
        },
        'budget': {  # RTX 4070, 3070
            'batch_size': 8,
            'gradient_accumulation_steps': 8,
            'fp16': True,
            'gradient_checkpointing': True,
            'dataloader_num_workers': 2
        }
    }
    
    if profile_name in profiles:
        profile = profiles[profile_name]
        trainer.config['training'].update(profile)
        logger.info(f"Applied GPU profile: {profile_name}")
    else:
        logger.warning(f"Unknown profile: {profile_name}")

# GPUTrainerクラスにプロファイル適用メソッドを追加
GPUTrainer._apply_gpu_profile = apply_gpu_profile


if __name__ == "__main__":
    exit(main())