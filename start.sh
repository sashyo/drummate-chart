#!/usr/bin/env bash
# DrumMate Chart - one-click start for Linux / macOS. First run installs everything into this folder.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${DRUMS_PORT:-8000}"
command -v python3 >/dev/null || { echo "Python 3.10+ is needed (https://www.python.org/downloads/)"; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null || { echo "Python 3.10+ is needed"; exit 1; }
if ! command -v ffmpeg >/dev/null; then
  if command -v brew >/dev/null; then echo "Installing ffmpeg with Homebrew..."; brew install ffmpeg
  elif command -v apt-get >/dev/null; then echo "Installing ffmpeg (sudo apt)..."; sudo apt-get install -y ffmpeg
  else echo "ffmpeg is needed: https://ffmpeg.org/download.html"; exit 1; fi
fi
if [ ! -x .venv/bin/python ]; then
  echo "Creating the Python environment and installing packages (first run only, 5-10 minutes, ~1.5 GB)..."
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
  if [ "$(uname)" = "Darwin" ]; then .venv/bin/pip install --quiet torch torchaudio
  else .venv/bin/pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu; fi
  .venv/bin/pip install --quiet demucs
fi
PY=.venv/bin/python; [ -x .venv-cuda/bin/python ] && PY=.venv-cuda/bin/python
command -v node >/dev/null || echo "Note: Node.js is not installed - YouTube links need it (https://nodejs.org); uploads and direct audio links work without it."
export DRUMS_LINKS="${DRUMS_LINKS:-1}" DRUMS_ALLOW_YOUTUBE="${DRUMS_ALLOW_YOUTUBE:-1}"
echo; echo "  DrumMate Chart -> http://127.0.0.1:$PORT   (Ctrl-C to stop; the first chart downloads the models, ~500 MB)"; echo
( sleep 2; (command -v xdg-open >/dev/null && xdg-open "http://127.0.0.1:$PORT") || (command -v open >/dev/null && open "http://127.0.0.1:$PORT") ) >/dev/null 2>&1 &
exec "$PY" -m uvicorn backend.server:app --host 127.0.0.1 --port "$PORT"
