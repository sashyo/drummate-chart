#!/usr/bin/env bash
# DrumMate Chart - one-click start for macOS / Linux.
# First run installs everything it needs (Python 3, ffmpeg, Node.js, the Python packages) into
# this folder or with the system package manager; later runs start in seconds.
set -uo pipefail
cd "$(dirname "$0")"
PORT="${DRUMS_PORT:-8000}"
OS="$(uname)"

say(){ echo; echo "  $*"; }

# --- package manager ---------------------------------------------------------
install_pkg() {   # install_pkg <mac-brew-name> <apt-name> <dnf-name> <pacman-name>
  if [ "$OS" = "Darwin" ]; then
    if ! command -v brew >/dev/null; then
      say "Installing Homebrew (it will ask for your password)..."
      NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || return 1
      eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
    fi
    brew install "$1"
  elif command -v apt-get >/dev/null; then sudo apt-get update -qq && sudo apt-get install -y "$2"
  elif command -v dnf >/dev/null; then sudo dnf install -y "$3"
  elif command -v pacman >/dev/null; then sudo pacman -S --noconfirm "$4"
  else say "Please install $1 yourself, then run this again."; return 1; fi
}

# --- prerequisites -------------------------------------------------------------
PYBIN=""
for c in python3.12 python3.11 python3.10 python3.13 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys,venv; assert sys.version_info>=(3,10)' 2>/dev/null; then PYBIN="$c"; break; fi
done
if [ -z "$PYBIN" ]; then
  say "Python 3.10+ is needed - installing it..."
  install_pkg python@3.12 "python3 python3-venv" python3 python || exit 1
  PYBIN="$(command -v python3.12 || command -v python3)"
fi
command -v ffmpeg >/dev/null || { say "ffmpeg is needed - installing it..."; install_pkg ffmpeg ffmpeg ffmpeg ffmpeg || exit 1; }
command -v node   >/dev/null || { say "Node.js is needed for YouTube links - installing it..."; install_pkg node nodejs nodejs nodejs || say "(continuing without Node: uploads and direct audio links work; YouTube links won't)"; }

# --- Python environment (first run only) -----------------------------------------
if [ ! -x .venv/bin/python ]; then
  say "Creating the Python environment and installing packages (first run only, 5-10 minutes, ~1.5 GB)..."
  "$PYBIN" -m venv .venv || exit 1
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt || exit 1
  if [ "$OS" = "Darwin" ]; then .venv/bin/pip install --quiet torch torchaudio || exit 1
  else .venv/bin/pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu || exit 1; fi
  .venv/bin/pip install --quiet demucs || exit 1
fi
PY=.venv/bin/python; [ -x .venv-cuda/bin/python ] && PY=.venv-cuda/bin/python

# --- run (links and YouTube allowed: your own machine, personal use) -------------
export DRUMS_LINKS="${DRUMS_LINKS:-1}" DRUMS_ALLOW_YOUTUBE="${DRUMS_ALLOW_YOUTUBE:-1}"
say "DrumMate Chart -> http://127.0.0.1:$PORT   (Ctrl-C to stop; the first chart downloads the models, ~500 MB)"; echo
if [ "${DRUMS_NO_BROWSER:-0}" != "1" ]; then
  ( sleep 2; (command -v xdg-open >/dev/null && xdg-open "http://127.0.0.1:$PORT") || (command -v open >/dev/null && open "http://127.0.0.1:$PORT") ) >/dev/null 2>&1 &
fi
exec "$PY" -m uvicorn backend.server:app --host 127.0.0.1 --port "$PORT"
