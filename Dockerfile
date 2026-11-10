FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY transcriber.py app.py ./
COPY static ./static

# The model (~9 GB) is not baked in: mount it at /model (see docker-compose.yml).
# Without a mount, set ASR_MODEL to the repo id and it downloads into HF_HOME.
ENV ASR_MODEL=/model \
    DATA_DIR=/data \
    HF_HOME=/hf-cache \
    HOST=0.0.0.0 \
    PORT=8765 \
    PYTHONUNBUFFERED=1
EXPOSE 8765
CMD ["python", "app.py"]
