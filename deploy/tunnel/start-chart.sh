#!/usr/bin/env bash
# Serve DrumMate Chart from this machine on chart.drummate.app.
# Run after every reboot (or wire into Task Scheduler / wsl.conf).
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1. the app itself on :8000
if ! curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  PY=.venv/bin/python; [ -x .venv-cuda/bin/python ] && PY=.venv-cuda/bin/python
  # the public site takes uploads only; a private/local build may set DRUMS_LINKS=1 (and DRUMS_ALLOW_YOUTUBE=1)
  DRUMS_LINKS=${DRUMS_LINKS:-0} DRUMS_ALLOW_YOUTUBE=${DRUMS_ALLOW_YOUTUBE:-0} DRUMS_YOUTUBE_WITH_CONSENT=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup $PY -m uvicorn backend.server:app \
    --host 127.0.0.1 --port 8000 >> /tmp/drummate-chart.log 2>&1 < /dev/null &
  echo "server starting on :8000"
  for i in $(seq 1 40); do sleep 0.5; curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1 && break; done
fi

# 2. the tunnel
if [ -f "$HOME/.cloudflared/chart-token" ]; then
  # dashboard-managed tunnel (connector token pasted into that file)
  setsid nohup "$HOME/.local/bin/cloudflared" tunnel run \
    --token "$(cat "$HOME/.cloudflared/chart-token")" \
    >> /tmp/drummate-tunnel.log 2>&1 < /dev/null &
else
  # locally-managed tunnel (cert.pem from `cloudflared tunnel login`)
  setsid nohup "$HOME/.local/bin/cloudflared" tunnel \
    --config "$HOME/.cloudflared/chart-config.yml" run drummate-chart \
    >> /tmp/drummate-tunnel.log 2>&1 < /dev/null &
fi
sleep 3
echo "tunnel log:"; tail -4 /tmp/drummate-tunnel.log
echo
echo "  https://chart.drummate.app"
