# src/models/gpu_optimized_model.py
"""
GPU最適化されたMIDI Composer RAGモデル
CUDA 12.x対応完全版
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import PreTrainedModel, PretrainedConfig
import math
from typing import Optional, Tuple, Dict, Any, List
import logging
import os
import yaml

logger = logging.getLogger(__name__)

class MIDIComposerRAGConfig(PretrainedConfig):
    """MIDI Composer RAG設定クラス"""
    
    model_type = "midi_composer_rag"
    
    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
        composer_embedding_dim: int = 64,
        feature_dim: int = 256,
        use_flash_attention: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.composer_embedding_dim = composer_embedding_dim
        self.feature_dim = feature_dim
        self.use_flash_attention = use_flash_attention

class GPUOptimizedMIDIRAG(PreTrainedModel):
    """GPU最適化されたMIDI RAGモデル"""
    
    config_class = MIDIComposerRAGConfig
    base_model_prefix = "midi_rag"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: MIDIComposerRAGConfig):
        super().__init__(config)
        self.config = config
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 混合精度設定
        self.use_amp = True
        self.scaler = GradScaler() if self.use_amp else None
        
        # Embedding層
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.composer_embeddings = nn.Embedding(10, config.composer_embedding_dim)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            OptimizedTransformerLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # 特徴量統合
        self.feature_projection = nn.Linear(config.feature_dim, config.hidden_size)
        
        # 出力層
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 重み初期化
        self.init_weights()
        
        # GPU最適化の適用
        self._apply_gpu_optimizations()
        
        logger.info(f"Model initialized on device: {self.device}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _apply_gpu_optimizations(self):
        """GPU最適化の適用"""
        if self.device.type == 'cuda':
            # PyTorch 2.0 Compile
            if hasattr(torch, 'compile'):
                try:
                    self.layers = torch.compile(self.layers)
                    logger.info("Model compiled with PyTorch 2.0")
                except Exception as e:
                    logger.warning(f"PyTorch compile failed: {e}")
            
            # cuDNN最適化
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        composer_ids: Optional[torch.LongTensor] = None,
        feature_context: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        **kwargs
    ):
        """順伝播"""
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Position IDs生成
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Hidden states初期化
        hidden_states = token_embeds + position_embeds
        
        # Composer embeddings追加
        if composer_ids is not None:
            composer_embeds = self.composer_embeddings(composer_ids)
            composer_embeds = composer_embeds.unsqueeze(1).expand(-1, seq_len, -1)
            # Composer情報を隠れ状態に統合
            hidden_states = hidden_states + self.feature_projection(
                torch.cat([
                    torch.zeros(batch_size, seq_len, self.config.feature_dim - self.config.composer_embedding_dim, device=device),
                    composer_embeds
                ], dim=-1)
            )
        
        # Feature context追加
        if feature_context is not None:
            feature_proj = self.feature_projection(feature_context)
            hidden_states = hidden_states + feature_proj.unsqueeze(1)
        
        # Attention mask生成
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Causal mask生成
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask
            )
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    @torch.no_grad()
    def generate_midi(
        self,
        prompt: str,
        composer_style: str = None,
        max_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ):
        """MIDI生成メソッド"""
        
        self.eval()
        
        # プロンプトをトークン化（簡易実装）
        input_ids = self._encode_prompt(prompt)
        batch_size = 1
        
        # 作曲家ID取得
        composer_id = self._get_composer_id(composer_style)
        composer_ids = torch.tensor([composer_id], device=self.device)
        
        # 生成ループ
        generated_tokens = input_ids.clone()
        
        for step in range(max_length - input_ids.size(1)):
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.forward(
                    input_ids=generated_tokens,
                    composer_ids=composer_ids
                )
                
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-k & Top-p sampling
                next_token = self._sample_next_token(logits, top_k, top_p)
                
                # トークン追加
                generated_tokens = torch.cat([
                    generated_tokens, 
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                
                # 終了条件チェック
                if next_token.item() == self.config.eos_token_id:
                    break
        
        # MIDIに変換
        midi = self._tokens_to_midi(generated_tokens[0])
        
        return midi
    
    def _encode_prompt(self, prompt: str) -> torch.LongTensor:
        """プロンプトをトークン化（簡易実装）"""
        # 実際の実装では適切なトークナイザーを使用
        start_token = 2
        tokens = [start_token] + [100, 101, 102]  # ダミートークン
        return torch.tensor([tokens], device=self.device)
    
    def _get_composer_id(self, composer_style: str) -> int:
        """作曲家名からIDを取得"""
        composer_map = {
            'chopin': 0, 'beethoven': 1, 'bach': 2, 'mozart': 3,
            'brahms': 4, 'liszt': 5, 'debussy': 6, 'rachmaninoff': 7
        }
        
        if composer_style:
            return composer_map.get(composer_style.lower(), 0)
        return 0
    
    def _sample_next_token(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """次のトークンをサンプリング"""
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # サンプリング
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token.squeeze(-1)
    
    def _tokens_to_midi(self, tokens: torch.LongTensor):
        """トークンをMIDIに変換（簡易実装）"""
        import pretty_midi
        
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        current_time = 0.0
        current_velocity = 80
        
        for token in tokens[1:]:  # skip start token
            token_id = token.item()
            
            if 4 <= token_id < 132:  # note tokens
                note_pitch = token_id - 4
                duration = 0.5
                
                note = pretty_midi.Note(
                    velocity=current_velocity,
                    pitch=note_pitch,
                    start=current_time,
                    end=current_time + duration
                )
                piano.notes.append(note)
                current_time += duration
            
            elif token_id == 3:  # end token
                break
        
        midi.instruments.append(piano)
        return midi


class OptimizedTransformerLayer(nn.Module):
    """GPU最適化されたTransformer層"""
    
    def __init__(self, config: MIDIComposerRAGConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        if config.use_flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self.attention = FlashMultiHeadAttention(config)
        else:
            self.attention = StandardMultiHeadAttention(config)
        
        # Feed forward network
        self.mlp = FusedMLP(config)
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None
    ):
        # Pre-norm + attention + residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        attn_output = self.attention(
            hidden_states, 
            attention_mask=attention_mask,
            causal_mask=causal_mask
        )
        
        hidden_states = residual + self.dropout(attn_output)
        
        # Pre-norm + MLP + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(mlp_output)
        
        return hidden_states


class FlashMultiHeadAttention(nn.Module):
    """Flash Attention実装"""
    
    def __init__(self, config: MIDIComposerRAGConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        assert self.hidden_size % self.num_heads == 0
        
        # QKV projection
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Flash Attention (PyTorch 2.0+)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Combine masks
            attn_mask = None
            if causal_mask is not None:
                attn_mask = causal_mask
                if attention_mask is not None:
                    # Expand attention mask to match causal mask shape
                    expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(
                        batch_size, 1, seq_len, seq_len
                    )
                    attn_mask = attn_mask.logical_and(expanded_mask)
            elif attention_mask is not None:
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(
                    batch_size, 1, seq_len, seq_len
                )
            
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=causal_mask is not None and attention_mask is None
            )
        else:
            # Fallback to standard attention
            attn_output = self._standard_attention(q, k, v, attention_mask, causal_mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        output = self.out_proj(attn_output)
        
        return output
    
    def _standard_attention(self, q, k, v, attention_mask, causal_mask):
        """標準的なattention実装（fallback）"""
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if causal_mask is not None:
            scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Apply attention mask
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~expanded_mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output


class StandardMultiHeadAttention(nn.Module):
    """標準的なMulti-Head Attention"""
    
    def __init__(self, config: MIDIComposerRAGConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None
    ):
        # PyTorchのMultiheadAttentionを使用
        attn_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask if attention_mask is not None else None,
            attn_mask=~causal_mask if causal_mask is not None else None,
            need_weights=False
        )
        
        return attn_output


class FusedMLP(nn.Module):
    """Fused MLP (Feed Forward Network)"""
    
    def __init__(self, config: MIDIComposerRAGConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.hidden_size * 4
        
        # Linear layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Activation
        self.activation = nn.SiLU()  # Swish activation
    
    def forward(self, hidden_states: torch.Tensor):
        # SwiGLU activation pattern
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        
        # Apply activation to gate and element-wise multiply with up
        intermediate = self.activation(gate) * up
        
        # Down projection
        output = self.down_proj(intermediate)
        
        return output


# GPU監視とプロファイリング
class GPUProfiler:
    """GPU使用状況プロファイラー"""
    
    @staticmethod
    def log_memory_usage(stage: str = ""):
        """メモリ使用量をログ出力"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            logger.info(f"GPU Memory {stage}: "
                       f"Allocated={allocated:.2f}GB, "
                       f"Reserved={reserved:.2f}GB, "
                       f"Max={max_allocated:.2f}GB")
    
    @staticmethod
    def clear_cache():
        """GPUキャッシュをクリア"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    @staticmethod
    def reset_peak_stats():
        """ピーク統計をリセット"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


# モデルファクトリー
class MIDIRAGModelFactory:
    """モデル作成ファクトリークラス"""
    
    @staticmethod
    def create_model(config_path: str = None, **kwargs) -> GPUOptimizedMIDIRAG:
        """設定に基づいてモデルを作成"""
        
        # デフォルト設定
        default_config = {
            'vocab_size': 512,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'num_hidden_layers': 12,
            'max_position_embeddings': 2048,
            'dropout': 0.1,
            'use_flash_attention': True
        }
        
        # 設定ファイルがある場合は読み込み
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config.get('model', {}))
        
        # kwargs で上書き
        default_config.update(kwargs)
        
        # 設定オブジェクト作成
        config = MIDIComposerRAGConfig(**default_config)
        
        # モデル作成
        model = GPUOptimizedMIDIRAG(config)
        
        # GPU最適化適用
        if torch.cuda.is_available():
            model = model.cuda()
            
            # 混合精度対応
            if default_config.get('use_fp16', True):
                model = model.half()
        
        return model
    
    @staticmethod
    def get_model_info(model: GPUOptimizedMIDIRAG) -> Dict[str, Any]:
        """モデル情報を取得"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # FP32 assuming
            'device': str(model.device),
            'dtype': str(next(model.parameters()).dtype),
            'config': model.config.to_dict()
        }


# メイン実行部分
def main():
    """メイン関数（テスト用）"""
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # GPU確認
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("GPU not available, using CPU")
    
    # モデル作成
    model = MIDIRAGModelFactory.create_model()
    
    # モデル情報表示
    info = MIDIRAGModelFactory.get_model_info(model)
    print(f"\nModel Info:")
    for key, value in info.items():
        if key != 'config':
            print(f"  {key}: {value}")
    
    # 簡単なテスト
    try:
        # ダミー入力
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, 512, (batch_size, seq_len))
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # GPUメモリ使用量記録
        GPUProfiler.log_memory_usage("Before Forward")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        GPUProfiler.log_memory_usage("After Forward")
        
        print(f"\nTest Results:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output logits shape: {outputs['logits'].shape}")
        print("✅ Model test passed!")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    finally:
        # クリーンアップ
        GPUProfiler.clear_cache()


if __name__ == "__main__":
    main()