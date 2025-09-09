#!/bin/bash
#
# 🚀 MIDI Composer RAG GPU最適化 完全セットアップスクリプト
# CUDA 12.x 対応版
#

set -e

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# バナー表示
show_banner() {
    echo -e "${CYAN}"
    echo "████████████████████████████████████████████████████████████"
    echo "█                                                          █"
    echo "█        🎵 MIDI Composer RAG GPU OPTIMIZER 🚀           █"
    echo "█                                                          █"
    echo "█     Classical Music AI with CUDA 12 Acceleration        █"
    echo "█                                                          █"
    echo "████████████████████████████████████████████████████████████"
    echo -e "${NC}"
    echo
}

# 進行状況表示
progress_bar() {
    local progress=$1
    local total=100
    local width=50
    local percentage=$((progress * 100 / total))
    local completed=$((progress * width / total))
    local remaining=$((width - completed))
    
    printf "\r${BLUE}Progress: [${GREEN}"
    printf "%*s" $completed | tr ' ' '='
    printf "${YELLOW}"
    printf "%*s" $remaining | tr ' ' '-'
    printf "${BLUE}] ${percentage}%%${NC}"
    
    if [ $progress -eq $total ]; then
        echo
    fi
}

# GPU環境確認
check_gpu_environment() {
    echo -e "${BLUE}🔍 GPU環境を確認中...${NC}"
    
    # NVIDIA GPU確認
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}❌ nvidia-smi が見つかりません。NVIDIAドライバーをインストールしてください${NC}"
        exit 1
    fi
    
    # GPU情報表示
    echo -e "${PURPLE}検出されたGPU:${NC}"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
    
    # CUDA確認
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo -e "${GREEN}✅ CUDA Version: $cuda_version${NC}"
        
        if [[ "$cuda_version" =~ ^12\. ]]; then
            echo -e "${GREEN}✅ CUDA 12.x が検出されました${NC}"
        else
            echo -e "${YELLOW}⚠️  CUDA 12.x以外のバージョンです: $cuda_version${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  nvcc が見つかりません${NC}"
    fi
    
    echo
}

# システム依存関係インストール
install_system_dependencies() {
    echo -e "${BLUE}📦 システム依存関係をインストール中...${NC}"
    
    progress_bar 10
    sudo apt update -qq
    
    progress_bar 30
    sudo apt install -y \
        python3 python3-pip python3-venv \
        build-essential cmake pkg-config \
        libasound2-dev jackd2 qjackctl \
        fluidsynth timidity \
        libsndfile1-dev libfftw3-dev portaudio19-dev \
        ffmpeg git wget curl > /dev/null 2>&1
    
    progress_bar 100
    echo -e "${GREEN}✅ システム依存関係インストール完了${NC}"
}

# Python仮想環境セットアップ
setup_python_env() {
    echo -e "${BLUE}🐍 Python環境をセットアップ中...${NC}"
    
    # 仮想環境作成
    if [ ! -d "midi_rag_env" ]; then
        python3 -m venv midi_rag_env
    fi
    
    # 仮想環境をアクティベート
    source midi_rag_env/bin/activate
    
    # pipアップグレード
    pip install -q --upgrade pip setuptools wheel
    
    echo -e "${GREEN}✅ Python環境セットアップ完了${NC}"
}

# GPU対応Pythonパッケージインストール
install_gpu_packages() {
    echo -e "${BLUE}⚡ GPU対応パッケージをインストール中...${NC}"
    
    source midi_rag_env/bin/activate
    
    progress_bar 20
    # PyTorch (CUDA 12.1対応)
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    progress_bar 40
    # GPU最適化ライブラリ
    pip install -q accelerate bitsandbytes flash-attn faiss-gpu
    
    progress_bar 60
    # 音楽処理ライブラリ
    pip install -q pretty-midi music21 mido python-rtmidi librosa madmom
    
    progress_bar 80
    # 機械学習・RAG
    pip install -q transformers sentence-transformers chromadb
    pip install -q numpy pandas scikit-learn matplotlib seaborn
    
    # Web UI
    pip install -q fastapi uvicorn streamlit gradio
    
    # HuggingFace
    pip install -q huggingface_hub safetensors
    
    # GPU監視
    pip install -q gpustat py3nvml psutil
    
    # ユーティリティ
    pip install -q tqdm jupyter python-dotenv pyyaml wandb
    
    progress_bar 100
    echo -e "${GREEN}✅ GPU対応パッケージインストール完了${NC}"
}

# プロジェクト構造作成
create_project_structure() {
    echo -e "${BLUE}📁 プロジェクト構造を作成中...${NC}"
    
    local project_name="midi_rag_project"
    mkdir -p $project_name
    cd $project_name
    
    # ディレクトリ構造作成
    mkdir -p {data/{raw/{chopin,beethoven,bach,mozart},processed,embeddings,models},src/{models/{feature_extraction,embedding,generation},utils,api},config,logs,scripts,output/{generated_midi,audio},cache,checkpoints}
    
    # 基本ファイル作成
    touch requirements.txt README.md .env .gitignore
    touch src/__init__.py src/models/__init__.py src/utils/__init__.py src/api/__init__.py
    
    echo -e "${GREEN}✅ プロジェクト構造作成完了${NC}"
}

# 設定ファイル作成
create_config_files() {
    echo -e "${BLUE}⚙️  設定ファイルを作成中...${NC}"
    
    # メイン設定ファイル
    cat > config/config.yaml << 'EOF'
# MIDI RAG システム設定

# データベース設定
database:
  chromadb_path: "./data/chromadb"
  collection_name: "midi_composers"

# 特徴量設定
features:
  embedding_dim: 256
  max_sequence_length: 1024

# モデル設定
model:
  vocab_size: 512
  hidden_size: 768
  num_attention_heads: 12
  num_hidden_layers: 12
  max_position_embeddings: 2048
  dropout: 0.1

# GPU設定
gpu:
  enabled: true
  device: "cuda"
  mixed_precision: true
  compile_model: true

# 学習設定
training:
  batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  weight_decay: 0.01
  max_grad_norm: 1.0
  warmup_ratio: 0.1
  fp16: true
  gradient_checkpointing: true
  dataloader_num_workers: 4
  pin_memory: true

# 生成設定
generation:
  max_length: 1024
  temperature: 0.8
  top_p: 0.9
  top_k: 50
  use_cache: true

# 監視設定
monitoring:
  enabled: true
  log_interval: 10
  gpu_stats: true
EOF

    # 環境変数設定
    cat > .env << 'EOF'
# 環境変数設定
PYTHONPATH=./src
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
HF_HOME=./cache/huggingface
TRANSFORMERS_CACHE=./cache/transformers
TOKENIZERS_PARALLELISM=true
EOF

    # requirements.txt
    cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
bitsandbytes>=0.39.0
flash-attn>=2.0.0
pretty-midi>=0.2.10
music21>=8.1.0
chromadb>=0.4.0
faiss-gpu>=1.7.4
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
streamlit>=1.25.0
gradio>=3.40.0
fastapi>=0.100.0
uvicorn>=0.23.0
huggingface_hub>=0.15.0
safetensors>=0.3.0
gpustat>=1.1.0
py3nvml>=0.2.7
psutil>=5.9.0
tqdm>=4.65.0
python-dotenv>=1.0.0
pyyaml>=6.0
wandb>=0.15.0
EOF

    echo -e "${GREEN}✅ 設定ファイル作成完了${NC}"
}

# GPUプロファイル検出
detect_gpu_profile() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "cpu"
        return
    fi
    
    local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr '[:upper:]' '[:lower:]')
    
    if [[ "$gpu_name" =~ (a100|h100) ]]; then
        echo "enterprise"
    elif [[ $gpu_memory -gt 20000 ]]; then
        echo "high_end"
    elif [[ $gpu_memory -gt 12000 ]]; then
        echo "mainstream"
    elif [[ $gpu_memory -gt 8000 ]]; then
        echo "budget"
    else
        echo "low_memory"
    fi
}

# システムテスト
run_system_tests() {
    echo -e "${BLUE}🧪 システムテストを実行中...${NC}"
    
    source midi_rag_env/bin/activate
    
    # PyTorchテスト
    python3 -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"
    
    # ライブラリテスト
    python3 -c "
try:
    import transformers, accelerate
    import pretty_midi, music21, chromadb
    import streamlit, gradio, fastapi
    print('✅ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"
    
    echo -e "${GREEN}✅ システムテスト完了${NC}"
}

# メイン実行関数
main() {
    show_banner
    
    # 引数解析
    SETUP_MODE="full"
    
    case "${1:-}" in
        --full)
            SETUP_MODE="full"
            ;;
        --quick)
            SETUP_MODE="quick"
            ;;
        --help|-h)
            echo "Usage: $0 [--full|--quick|--help]"
            echo "  --full   Complete setup with all dependencies"
            echo "  --quick  Quick setup with minimal dependencies"
            echo "  --help   Show this help message"
            exit 0
            ;;
    esac
    
    check_gpu_environment
    
    local gpu_profile=$(detect_gpu_profile)
    echo -e "${PURPLE}🎯 Detected GPU profile: ${gpu_profile}${NC}"
    
    # 実行確認
    echo -e "${YELLOW}Setup mode: ${SETUP_MODE}${NC}"
    echo
    read -p "Continue with GPU-optimized setup? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    
    echo -e "\n${CYAN}🚀 Starting GPU-optimized MIDI RAG setup...${NC}\n"
    
    # セットアップ実行
    case $SETUP_MODE in
        "full")
            install_system_dependencies
            setup_python_env
            install_gpu_packages
            create_project_structure
            create_config_files
            run_system_tests
            ;;
        "quick")
            setup_python_env
            install_gpu_packages
            create_project_structure
            create_config_files
            ;;
    esac
    
    # 完了メッセージ
    echo -e "\n${GREEN}🎉 GPU-optimized MIDI Composer RAG setup completed!${NC}\n"
    
    echo -e "${CYAN}Quick Start Commands:${NC}"
    echo "📁 cd midi_rag_project"
    echo "🔋 source ../midi_rag_env/bin/activate"
    echo "🌐 streamlit run src/api/web_interface.py"
    echo
    
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Place MIDI files in data/raw/{composer}/ directories"
    echo "2. Set HuggingFace token: export HF_TOKEN='your_token'"
    echo "3. Run training or use the web interface"
    echo
}

# スクリプト実行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi