# chordcodes



bash# 環境セットアップ
python run.py setup --gpu

# モデル学習（GPU最適化）
python run.py train --epochs 10 --gpu --wandb

# 音楽生成
python run.py generate --interactive

# Webインターフェース起動
python run.py web --interface streamlit

HuggingFaceアップロード

# 環境変数設定
export HF_TOKEN="your_huggingface_token"

# アップロード実行
python run.py upload --model-name midi-composer-rag-gpu


