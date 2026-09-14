FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --registry=https://registry.npmjs.org
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    SESSION_DB=/app/data/sessions.db \
    CHECKPOINT_DB=/app/data/checkpoints.db \
    OCR_TESSDATA_DIR=/opt/rolepilot-ocr/tessdata \
    TESSERACT_CMD=tesseract \
    OCR_LANGUAGES=chi_sim+eng \
    TIKTOKEN_CACHE_DIR=/opt/rolepilot-tokenizer
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates tesseract-ocr tesseract-ocr-eng fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements.lock.txt
COPY app/ ./app/
COPY api/ ./api/
COPY assets/ocr/ /opt/rolepilot-ocr/
COPY scripts/install_ocr_languages.py scripts/ocr_smoke.py ./scripts/
COPY LICENSE THIRD_PARTY_NOTICES.md ./
COPY --from=frontend /build/frontend/dist ./frontend/dist
RUN python -c "from app.parsers.ocr import dependency_status; s=dependency_status(); assert s['ready'] and s['pdf']['ready'], s" \
    && python -c "import tiktoken; tiktoken.get_encoding('o200k_base')" \
    && useradd --uid 10001 --create-home rolepilot \
    && mkdir -p /app/data \
    && chown rolepilot:rolepilot /app/data
USER rolepilot
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
