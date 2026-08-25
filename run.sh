#!/usr/bin/env bash
# Start the Drum Notation app.  Usage: ./run.sh [port]
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8000}"
VENV=".venv"
# a CUDA-capable environment (see docs: .venv-cuda) is preferred when present
[ -x ".venv-cuda/bin/python" ] && VENV=".venv-cuda"

if [ ! -d "$VENV" ]; then
  echo "→ creating virtualenv"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  echo "→ installing dependencies (this takes a few minutes the first time)"
  "$VENV/bin/pip" install --quiet -r requirements.txt
  echo "→ installing Demucs for drum separation (large, optional)"
  "$VENV/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cpu || \
    echo "  (torch failed — the app will fall back to the fast percussive filter)"
  "$VENV/bin/pip" install --quiet demucs || true
fi

command -v ffmpeg >/dev/null || { echo "ffmpeg is required: apt install ffmpeg"; exit 1; }

echo
echo "  DrumMate Chart → http://127.0.0.1:${PORT}"
echo
exec "$VENV/bin/python" -m uvicorn backend.server:app --host 127.0.0.1 --port "$PORT"
