FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    DATA_DIR=/app/data/app \
    UPLOAD_DIR=/app/data/app/uploads \
    GENERATED_DIR=/app/data/app/generated \
    DATABASE_PATH=/app/data/app/vidquery.sqlite3 \
    ENABLE_YOLO=true \
    ENABLE_WHISPER=true \
    ENABLE_DIARIZATION=true \
    REQUIRE_DIARIZATION=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG VIDQUERY_EXTRAS=full
COPY requirements.txt pyproject.toml README.md ./
COPY vidquery ./vidquery
COPY frontend ./frontend
RUN pip install --no-cache-dir ".[${VIDQUERY_EXTRAS}]"
RUN if [ "${VIDQUERY_EXTRAS}" = "full" ]; then \
      pip install --no-cache-dir --force-reinstall \
        torch==2.5.1+cpu torchaudio==2.5.1+cpu torchvision==0.20.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi
COPY . .

EXPOSE 8000
CMD ["python", "-m", "vidquery.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
