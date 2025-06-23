FROM python:3.11-slim

WORKDIR /app

# dtw-python のビルドに必要な gcc などをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY test.py ./
COPY inputs/ ./inputs/

CMD ["python", "test.py"]
