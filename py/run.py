# run.py
#!/usr/bin/env python3
"""
🎵 MIDI Composer RAG - メインランチャー
GPU最適化対応完全版

使用方法:
  python run.py setup              # 環境セットアップ
  python run.py train              # モデル学習
  python run.py generate           # 音楽生成
  python run.py web               # Webインターフェース起動
  python run.py upload            # HuggingFaceアップロード
  python run.py --help            # ヘルプ表示
"""

import os
import sys
import argparse
import subprocess
import logging
import time
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MIDIRAGLauncher:
    """MIDI RAG メインランチャー"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_path = self.project_root / "src"
        self.config_path = self.project_root / "config"
        
        # パスを追加
        if str(self.src_path) not in sys.path:
            sys.path.append(str(self.src_path))
    
    def setup_environment(self, gpu: bool = True, profile: str = None):
        """環境セットアップ"""
        logger.info("🛠️ Setting up MIDI RAG environment...")
        
        try:
            # GPU最適化設定生成
            if gpu and torch.cuda.is_available():
                from utils.gpu_optimizer import auto_setup_gpu_config
                
                config_path = self.config_path / f"{profile or 'auto'}_gpu_config.yaml"
                auto_setup_gpu_config(str(config_path))
                logger.info(f"✅ GPU config created: {config_path}")
                
                # システム情報表示
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                logger.warning("⚠️ GPU not available or disabled")
            
            # 必要なディレクトリ作成
            directories = [
                "data/raw/chopin", "data/raw/beethoven", "data/raw/bach", "data/raw/mozart",
                "data/processed", "data/embeddings", "checkpoints", "logs", "output"
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ Environment setup completed!")
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            raise
    
    def train_model(self, 
                   config: str = None,
                   epochs: int = 10,
                   gpu: bool = True, 
                   profile: str = None,
                   resume: str = None,
                   wandb: bool = False):
        """モデル学習"""
        logger.info("🎓 Starting model training...")
        
        try:
            # デフォルト設定ファイル
            if not config:
                if gpu and torch.cuda.is_available():
                    config = str(self.config_path / "auto_gpu_config.yaml")
                else:
                    config = str(self.config_path / "config.yaml")
            
            # 学習スクリプト実行
            cmd = [
                sys.executable, 
                str(self.src_path / "train_gpu.py"),
                "--config", config,
                "--epochs", str(epochs)
            ]
            
            if profile:
                cmd.extend(["--profile", profile])
            
            if resume:
                cmd.extend(["--resume", resume])
            
            if wandb:
                cmd.append("--wandb")
            
            # Dry runオプション
            if self._is_dry_run():
                cmd.append("--dry_run")
                logger.info("🧪 Running in dry-run mode")
            
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            
            if result.returncode == 0:
                logger.info("🎉 Training completed successfully!")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Training failed with exit code {e.returncode}")
            raise
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            raise
    
    def generate_music(self, 
                      prompt: str = None,
                      composer: str = "Chopin",
                      output: str = None,
                      model_path: str = None,
                      interactive: bool = False):
        """音楽生成"""
        logger.info("🎵 Generating music...")
        
        try:
            if interactive:
                # インタラクティブモード
                self._interactive_generation()
            else:
                # コマンドライン生成
                if not prompt:
                    prompt = input("Enter generation prompt: ")
                
                if not output:
                    output = f"generated_{composer.lower()}_{int(time.time())}.mid"
                
                if not model_path:
                    model_path = str(Path("checkpoints/best_model.pt"))
                
                # 生成スクリプト実行
                cmd = [
                    sys.executable,
                    str(self.src_path / "generate.py"),
                    "--prompt", prompt,
                    "--composer", composer,
                    "--output", output
                ]
                
                if os.path.exists(model_path):
                    cmd.extend(["--model_path", model_path])
                
                logger.info(f"Executing: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                
                logger.info(f"✅ Music generated: {output}")
        
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise
    
    def _interactive_generation(self):
        """インタラクティブ生成"""
        print("\\n🎼 Interactive Music Generation")
        print("=" * 50)
        
        while True:
            try:
                print("\\nOptions:")
                print("1. Generate with prompt")
                print("2. Use preset prompt") 
                print("3. Batch generation")
                print("4. Exit")
                
                choice = input("\\nSelect option (1-4): ").strip()
                
                if choice == "1":
                    prompt = input("Enter prompt: ")
                    composer = input("Composer (Chopin/Beethoven/Bach/Mozart): ") or "Chopin"
                    
                    self.generate_music(prompt=prompt, composer=composer)
                
                elif choice == "2":
                    presets = {
                        "1": "Generate a romantic nocturne with expressive melodies",
                        "2": "Create a dramatic sonata movement with powerful themes", 
                        "3": "Compose a baroque fugue with intricate counterpoint",
                        "4": "Write a classical minuet with elegant phrasing"
                    }
                    
                    print("\\nPreset prompts:")
                    for key, value in presets.items():
                        print(f"{key}. {value}")
                    
                    preset_choice = input("Select preset (1-4): ").strip()
                    if preset_choice in presets:
                        prompt = presets[preset_choice]
                        composer = ["Chopin", "Beethoven", "Bach", "Mozart"][int(preset_choice)-1]
                        
                        self.generate_music(prompt=prompt, composer=composer)
                
                elif choice == "3":
                    self._batch_generation()
                
                