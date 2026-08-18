# Deploying chart.drummate.app

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
