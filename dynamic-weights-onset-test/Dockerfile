FROM python:3.10-slim

# 必要なツールとライブラリのインストール
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ作成
WORKDIR /app

# requirementsの準備（後でキャッシュ活用のため）
COPY requirements.txt .

# Pythonライブラリのインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルのコピー
COPY . .

# 実行コマンド
CMD ["python", "weightsOnset.py"]
