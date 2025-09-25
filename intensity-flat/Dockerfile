FROM python:3.11-slim

# ランタイム依存（libsndfile は soundfile 用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY flatten_envelope.py /app/flatten_envelope.py
ENTRYPOINT ["python", "/app/flatten_envelope.py"]