FROM python:3.11-slim

LABEL description="Multi-Agent Employee Operations Copilot"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HOME=/home/copilot \
    HF_HOME=/home/copilot/.cache/huggingface \
    ANONYMIZED_TELEMETRY=False \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN groupadd --gid 10001 copilot \
    && useradd --uid 10001 --gid 10001 \
       --create-home copilot

COPY --chown=copilot:copilot . .

RUN mkdir -p \
    /app/data/checkpoints \
    /app/data/logs \
    /app/data/vector_store \
    /app/data/evaluation/results \
    /home/copilot/.cache/huggingface \
    && chown -R copilot:copilot \
       /app /home/copilot

USER copilot

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s \
    --start-period=90s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
