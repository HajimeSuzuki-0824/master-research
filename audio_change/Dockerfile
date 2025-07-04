FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# dtw-python のビルドに必要な gcc などをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY inputs ./
COPY apis ./

CMD ["python", "main.py"]
