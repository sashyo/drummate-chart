# Deploying chart.drummate.app

> Migrating to another machine? Follow **[../DEPLOY.md](../DEPLOY.md)** — the step-by-step runbook
> (tunnel identity, GPU environment, data restore, verification, cutover, autostart).

Two pieces, mirroring the rest of the DrumMate family:

## 1. The subdomain Worker (frontend + API proxy)

From this repository root, with the drummate repo's wrangler:

    ~/drummate/relay/node_modules/.bin/wrangler deploy --config deploy/worker/wrangler.jsonc
    ~/drummate/relay/node_modules/.bin/wrangler secret put BACKEND_ORIGIN --config deploy/worker/wrangler.jsonc

`BACKEND_ORIGIN` is the URL of the backend below. Until it is set the site
serves but /api/* answers 503 with a friendly message.

## 2. The backend (FastAPI + Demucs + ffmpeg — needs real CPU)

Anything that runs a container works; the image is defined in ../Dockerfile.

    docker compose up -d          # on a VPS
    # or
    fly launch --no-deploy && fly deploy   # ~4GB RAM recommended (shared-cpu-2x)

Caveats worth knowing before going public:
- yt-dlp from a datacenter IP often trips YouTube's bot checks; set
  DRUMS_COOKIE_FILE to a cookies.txt export if that happens.
- One transcription at a time is enforced in-process; a 3-minute song takes
  roughly 1.5-2 minutes of full CPU. Size the machine accordingly.
- DRUMS_MAX_SECONDS (default 600) caps how much audio is analysed per job.

## Live deployment (current): this PC + Cloudflare Tunnel

The site is served at https://chart.drummate.app through a locally-managed
Cloudflare Tunnel (`drummate-chart`, credentials in `~/.cloudflared/`) that
points at the FastAPI server on localhost:8000. No Worker involved.

After a reboot (WSL does not keep services alive):

    ./deploy/tunnel/start-chart.sh

It starts the app if needed and reconnects the tunnel. Logs:
`/tmp/drummate-chart.log` and `/tmp/drummate-tunnel.log`.

The app is public - consider gating it with Cloudflare Access (Zero Trust ->
Access -> Applications, free tier) behind an email one-time code.
