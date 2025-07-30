FROM python:3.9-slim

ENV PYTHONBUFFERED=1

# システムの依存関係をインストール
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    libsndfile1-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 要求ファイルをコピーして依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトファイルをコピー
COPY . .

# 出力ディレクトリを作成
RUN mkdir -p outputs

# デフォルトコマンド
CMD ["python", "main.py"]