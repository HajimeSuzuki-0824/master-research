FROM python:3.11-slim

WORKDIR /app

# システム依存ライブラリのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Pythonライブラリのインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイル一式をコピー
COPY main.py ./
COPY api/ ./api/
COPY sounds/ ./sounds/
COPY png/ ./png/
