# DrumMate chart transcriber — backend + frontend in one container.
# CPU-only torch keeps the image lean enough; Demucs runs fine on CPU.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir demucs

COPY backend ./backend
COPY frontend ./frontend

# Pre-download the separation model so the first user isn't the one waiting on it.
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

ENV DRUMS_DATA=/data
VOLUME /data
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
