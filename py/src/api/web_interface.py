# src/api/web_interface.py
#!/usr/bin/env python3
"""
MIDI Composer RAG - Streamlit Web Interface
GPU最適化対応完全版
"""

import streamlit as st
import torch
import sys
import os
import tempfile
import time
import yaml
from pathlib import Path
import logging

# カスタムモジュール
sys.path.append('src')

try:
    from models.gpu_optimized_model import GPUOptimizedMIDIRAG, MIDIRAGModelFactory, GPUProfiler
    from utils.gpu_optimizer import GPUOptimizer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure all modules are properly installed and src/ is in the path")
    st.stop()

# ページ設定
st.set_page_config(
    page_title="🎵 MIDI Composer RAG",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.gpu-info {
    background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
    padding: 0.5rem;
    border-radius: 5px;
    color: white;
    margin: 0.5rem 0;
}

.composer-card {
    background: #f8f9ff;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}

.generation-stats {
    background: #fff3cd;
    padding: 1rem;
    border-radius: 5px;
    border: 1px solid #ffeaa7;
}

.error-box {
    background: #f8d7da;
    padding: 1rem;
    border-radius: 5px;
    border: 1px solid #f5c6cb;
    color: #721c24;
}

.success-box {
    background: #d4edda;
    padding: 1rem;
    border-radius: 5px;
    border: 1px solid #c3e6cb;
    color: #155724;
}
</style>
""", unsafe_allow_html=True)

# セッション状態初期化
def initialize_session_state():
    """セッション状態を初期化"""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.gpu_optimizer = None
        st.session_state.generation_history = []
        st.session_state.gpu_stats = {}

initialize_session_state()

# GPU情報とシステム状態
@st.cache_data
def get_system_info():
    """システム情報取得"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'pytorch_version': torch.__version__,
        'python_version': sys.version.split()[0]
    }
    
    if torch.cuda.is_available():
        info.update({
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'cuda_version': torch.version.cuda
        })
    
    return info

@st.cache_resource
def load_model(model_path: str = "checkpoints/best_model.pt"):
    """モデルをキャッシュ付きで読み込み"""
    try:
        if os.path.exists(model_path):
            # チェックポイントからロード
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 設定からモデル作成
            model = MIDIRAGModelFactory.create_model()
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else:
            # デフォルトモデル作成
            st.warning(f"Model checkpoint not found at {model_path}. Using default model.")
            model = MIDIRAGModelFactory.create_model()
        
        # GPU最適化適用
        if torch.cuda.is_available():
            model = model.cuda()
            model.eval()
        
        return model, True
        
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, False

def get_gpu_memory_info():
    """GPU メモリ情報取得"""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
    }

# メインUI
def main():
    """メインUI"""
    
    # ヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>🎵 MIDI Composer RAG</h1>
        <p>Classical Music Generation with GPU Acceleration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # システム情報表示
        system_info = get_system_info()
        
        if system_info['cuda_available']:
            st.markdown(f"""
            <div class="gpu-info">
                🚀 <strong>GPU Accelerated</strong><br>
                📱 {system_info['gpu_name']}<br>
                💾 {system_info['gpu_memory_total']:.1f}GB VRAM
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No GPU detected - Running on CPU")
        
        st.markdown("---")
        
        # モデル設定
        st.subheader("🤖 Model Settings")
        
        model_path = st.text_input(
            "Model Path", 
            value="checkpoints/best_model.pt",
            help="Path to trained model checkpoint"
        )
        
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                model, success = load_model(model_path)
                if success:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.session_state.model_loaded = False
        
        st.markdown("---")
        
        # 生成設定
        st.subheader("🎼 Generation Settings")
        
        composer = st.selectbox(
            "Composer Style",
            options=["Chopin", "Beethoven", "Bach", "Mozart", "Brahms", "Liszt", "Debussy"],
            index=0,
            help="Select the composer style for generation"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "Temperature", 
                min_value=0.1, 
                max_value=2.0, 
                value=0.8, 
                step=0.1,
                help="Controls randomness in generation"
            )
        
        with col2:
            top_p = st.slider(
                "Top-p", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.9, 
                step=0.05,
                help="Nucleus sampling threshold"
            )
        
        max_length = st.slider(
            "Max Length", 
            min_value=64, 
            max_value=2048, 
            value=512, 
            step=64,
            help="Maximum sequence length"
        )
        
        # GPU最適化設定
        if system_info['cuda_available']:
            st.markdown("---")
            st.subheader("⚡ GPU Optimization")
            
            use_flash_attention = st.checkbox("Flash Attention", value=True)
            use_mixed_precision = st.checkbox("Mixed Precision", value=True)
            
            if st.button("🧹 Clear GPU Cache"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    st.success("GPU cache cleared!")
        
        # GPU使用状況
        if system_info['cuda_available']:
            st.markdown("---")
            st.subheader("📊 GPU Status")
            
            memory_info = get_gpu_memory_info()
            if memory_info:
                usage_percent = (memory_info['allocated_gb'] / memory_info['total_gb']) * 100
                
                st.metric(
                    "GPU Memory Usage", 
                    f"{memory_info['allocated_gb']:.1f} / {memory_info['total_gb']:.1f} GB",
                    f"{usage_percent:.1f}%"
                )
                
                # メモリ使用率バー
                st.progress(usage_percent / 100)
    
    # メインコンテンツ
    if not st.session_state.model_loaded:
        st.info("👆 Please load a model from the sidebar to start generating music!")
        
        # モデルなしでも使えるサンプル表示
        st.subheader("🎼 Example Generations")
        
        example_prompts = [
            {
                "title": "Romantic Nocturne",
                "prompt": "Generate a romantic Chopin-style nocturne in F major with expressive dynamics and flowing melodies",
                "composer": "Chopin"
            },
            {
                "title": "Dramatic Sonata", 
                "prompt": "Create a powerful Beethoven-style sonata movement with dramatic contrasts and development",
                "composer": "Beethoven"
            },
            {
                "title": "Baroque Fugue",
                "prompt": "Compose a Bach-style fugue with intricate counterpoint and mathematical precision", 
                "composer": "Bach"
            }
        ]
        
        for example in example_prompts:
            with st.container():
                st.markdown(f"""
                <div class="composer-card">
                    <h4>{example['title']} - {example['composer']}</h4>
                    <p><em>"{example['prompt']}"</em></p>
                </div>
                """, unsafe_allow_html=True)
        
        return
    
    # モデルロード済みの場合
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Generation Prompt")
        
        # プリセットプロンプト
        preset_prompts = {
            "Custom": "",
            "Romantic Nocturne": "Generate a romantic Chopin-style nocturne with expressive melodies and rich harmonies in a gentle, flowing tempo",
            "Dramatic Sonata": "Create a dramatic Beethoven-style sonata movement with powerful themes, dynamic contrasts, and developmental passages",
            "Baroque Fugue": "Compose a Bach-style fugue with intricate counterpoint, mathematical precision, and complex voice weaving",
            "Classical Elegance": "Write a Mozart-style piece with elegant phrasing, balanced structure, and graceful melodic lines",
            "Romantic Ballade": "Generate a Chopin-style ballade with narrative character, contrasting sections, and virtuosic passages",
            "Heroic Symphony": "Create a Beethoven-style symphonic movement with heroic themes and triumphant development"
        }
        
        selected_preset = st.selectbox(
            "💡 Preset Prompts",
            options=list(preset_prompts.keys()),
            index=0
        )
        
        if selected_preset != "Custom":
            default_prompt = preset_prompts[selected_preset]
        else:
            default_prompt = ""
        
        prompt = st.text_area(
            "Describe the music you want to generate:",
            value=default_prompt,
            height=120,
            placeholder="Enter a detailed description of the musical piece you want to generate...",
            help="Be specific about style, mood, tempo, key, and musical characteristics"
        )
        
        # 生成ボタン
        generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
        
        with generate_col2:
            if st.button("🎼 Generate Music", type="primary", use_container_width=True):
                if prompt.strip():
                    generate_music(prompt, composer, temperature, top_p, max_length)
                else:
                    st.warning("⚠️ Please enter a generation prompt")
    
    with col2:
        st.subheader("📊 Generation Status")
        
        # 生成状況表示エリア
        status_container = st.container()
        
        # 最近の生成履歴
        if st.session_state.generation_history:
            st.subheader("🕒 Recent Generations")
            
            for i, entry in enumerate(reversed(st.session_state.generation_history[-3:])):
                with st.expander(f"{entry['composer']} - {entry['timestamp']}"):
                    st.text(f"Prompt: {entry['prompt'][:100]}...")
                    st.text(f"Generation time: {entry['generation_time']:.2f}s")
                    
                    if entry['success']:
                        st.success("✅ Generated successfully")
                        
                        # ダウンロードボタン
                        if 'midi_data' in entry:
                            st.download_button(
                                "📥 Download MIDI",
                                data=entry['midi_data'],
                                file_name=f"generated_{entry['composer'].lower()}_{i}.mid",
                                mime="audio/midi"
                            )
                    else:
                        st.error(f"❌ Generation failed: {entry['error']}")

def generate_music(prompt: str, composer: str, temperature: float, top_p: float, max_length: int):
    """音楽生成処理"""
    
    # 生成開始時刻
    start_time = time.time()
    
    # ステータス表示
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    try:
        with status_placeholder:
            st.info("🎵 Generating music... Please wait")
        
        with progress_placeholder:
            progress_bar = st.progress(0)
            status_text = st.text("Initializing generation...")
        
        # GPU メモリ使用状況記録
        GPUProfiler.log_memory_usage("Before Generation")
        
        # 進行状況更新
        progress_bar.progress(0.2)
        status_text.text("Encoding prompt...")
        
        # モデルで生成
        model = st.session_state.model
        
        progress_bar.progress(0.4)
        status_text.text("Generating MIDI sequence...")
        
        with torch.no_grad():
            # 実際の生成処理
            midi = model.generate_midi(
                prompt=prompt,
                composer_style=composer.lower(),
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        
        progress_bar.progress(0.8)
        status_text.text("Converting to MIDI format...")
        
        # MIDI データをバイトに変換
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_file:
            midi.write(tmp_file.name)
            
            with open(tmp_file.name, "rb") as f:
                midi_data = f.read()
            
            os.unlink(tmp_file.name)
        
        progress_bar.progress(1.0)
        status_text.text("Generation completed!")
        
        # 生成完了
        generation_time = time.time() - start_time
        
        # GPU メモリ使用状況記録
        GPUProfiler.log_memory_usage("After Generation")
        
        # 成功メッセージ
        with status_placeholder:
            st.markdown("""
            <div class="success-box">
                ✅ <strong>Music generated successfully!</strong><br>
                🎼 Ready for download
            </div>
            """, unsafe_allow_html=True)
        
        with progress_placeholder:
            st.empty()
        
        # ダウンロードボタン
        st.download_button(
            label="📥 Download MIDI File",
            data=midi_data,
            file_name=f"generated_{composer.lower()}_{int(time.time())}.mid",
            mime="audio/midi",
            type="primary"
        )
        
        # 生成統計表示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Generation Time", f"{generation_time:.2f}s")
        
        with col2:
            st.metric("Sequence Length", max_length)
        
        with col3:
            if torch.cuda.is_available():
                memory_info = get_gpu_memory_info()
                st.metric("GPU Memory", f"{memory_info['allocated_gb']:.1f}GB")
        
        # 履歴に追加
        st.session_state.generation_history.append({
            'timestamp': time.strftime('%H:%M:%S'),
            'prompt': prompt,
            'composer': composer,
            'generation_time': generation_time,
            'success': True,
            'midi_data': midi_data
        })
        
        # GPU キャッシュクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    except Exception as e:
        # エラー処理
        generation_time = time.time() - start_time
        
        with status_placeholder:
            st.markdown(f"""
            <div class="error-box">
                ❌ <strong>Generation failed</strong><br>
                Error: {str(e)}
            </div>
            """, unsafe_allow_html=True)
        
        with progress_placeholder:
            st.empty()
        
        # エラー履歴に追加
        st.session_state.generation_history.append({
            'timestamp': time.strftime('%H:%M:%S'),
            'prompt': prompt,
            'composer': composer,
            'generation_time': generation_time,
            'success': False,
            'error': str(e)
        })
        
        # GPU キャッシュクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# フッター
def render_footer():
    """フッター表示"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("🚀 **GPU Accelerated**")
        st.caption("Powered by PyTorch & CUDA")
    
    with col2:
        st.markdown("🤗 **HuggingFace Ready**") 
        st.caption("Compatible with HF Hub")
    
    with col3:
        st.markdown("🎵 **Open Source**")
        st.caption("MIT License")

# アプリケーション実行
if __name__ == "__main__":
    try:
        main()
        render_footer()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check the logs and ensure all dependencies are installed.")
        
        # デバッグ情報
        if st.checkbox("Show debug info"):
            st.code(f"Error details:\n{e}")
            st.code(f"Python path: {sys.path}")
            st.code(f"Current directory: {os.getcwd()}")
            
            # システム情報
            system_info = get_system_info()
            st.json(system_info)
        '