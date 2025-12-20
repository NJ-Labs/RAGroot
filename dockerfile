FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    python3-pip \
    python3.11 \
    python3.11-dev \
    wget \
    # Tesseract OCR and language packs for Hebrew/English OCR
    tesseract-ocr \
    tesseract-ocr-heb \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

RUN mkdir -p /app/models /app/models/embeddings /models /data /index

COPY requirements.txt pyproject.toml* /app/
RUN --mount=type=cache,target=/root/.cache/pip /root/.local/bin/uv pip install --system -r requirements.txt

COPY main.py /app/
COPY .env.docker.example /app/
COPY Documentation/ /app/Documentation/
COPY static/ /app/static/
COPY tools/ /app/tools/
COPY utils/ /app/utils/
RUN cp /app/.env.docker.example /app/.env

ENV EMBEDDING_CACHE_DIR=/app/models/embeddings \
    EMBEDDING_LOCAL_ONLY=false \
    HF_HOME=/app/models/embeddings

RUN uv run python /app/tools/download_models.py --all

RUN if [ ! -f /app/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf ]; then \
    wget -q --show-progress \
    https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    -O /app/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf; \
    fi

RUN ln -sf /app/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf /models/llama-model.gguf && \
    ln -sf /app/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf /models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
# RUN uv run python -m utils.indexer
# RUN uv run python /app/tools/validate_offline.py || \
#     (echo "❌ Offline validation failed. Please check the build logs." && exit 1)

ENV DATA_PATH=/data/arxiv_2.9k.jsonl \
    INDEX_DIR=/index \
    MODEL_PATH=/models/llama-model.gguf \
    EMBEDDING_CACHE_DIR=/app/models/embeddings \
    EMBEDDING_LOCAL_ONLY=true \
    IMAGE_API_PROVIDER=pollinations \
    IMAGE_MODEL_PATH=/app/models/sdxl-turbo \
    IMAGE_MODEL_NAME=stabilityai/sdxl-turbo \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=0

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uv", "run", "python", "main.py"]