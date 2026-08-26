# Deploying / migrating chart.drummate.app

A runbook written so an AI coding assistant (or a person) can move the
live site to a new machine end to end. Facts, not vibes: every path,
version and command below is what the current deployment uses.

## What the site is

- **App**: FastAPI backend (`backend/server.py`) + static frontend (`frontend/`), one process, port 8000.
  It takes uploaded audio (links optional; the public site is upload-only), separates drums
  (Demucs, then the MDX23C DrumSep kit-splitter), derives tempo/grid from the drums, quantises,
  engraves, exports.
- **Public URL**: https://chart.drummate.app through a **locally-managed Cloudflare Tunnel**
  named `drummate-chart` that forwards to `http://localhost:8000` on the PC. No Worker,
  no reverse proxy, no open ports.
- **State on disk** (all under the repo unless `DRUMS_DATA` says otherwise):
  - `data/jobs/<id>/` — every chart ever made (score.json, MIDI, MusicXML, mp3 stems, job.json). ~1 GB.
  - `data/cache/` — downloaded audio + separation caches, keyed on audio content. Optional to migrate (30 GB); the app rebuilds it.
  - `data/stats.json` — the usage counter (`/api/stats`). Small; migrate it.
  - `data/learn/` — accuracy-research batches. Optional.
  - `~/.cache/drumsep/` — the kit-split model (437 MB `MDX23C-DrumSep-aufr33-jarredou.ckpt` + yaml). Auto-downloaded on first use; copying it saves a download.
  - Demucs weights auto-download into torch's hub cache on first run.

## 0. Before you start (on the OLD machine)

    cd ~/drum-notation && git status && git push          # nothing uncommitted
    tar czf ~/cloudflared-chart.tgz -C ~ .cloudflared/cert.pem .cloudflared/chart-config.yml \
        .cloudflared/<tunnel-id>.json
    # ^ the tunnel identity. Keep it private. (The other files in ~/.cloudflared belong to a different site.)
    tar czf ~/chart-data.tgz -C ~/drum-notation data/jobs data/stats.json    # charts + counter (~1 GB)
    tar czf ~/drumsep-model.tgz -C ~ .cache/drumsep                         # optional, saves a 437 MB download

Copy those three archives to the new machine (scp, USB, whatever).

## 1. New machine prerequisites

Tested on Ubuntu 24.04 (WSL2 or native). Needs:

| thing | version used | why |
|---|---|---|
| Python | 3.12 | backend |
| ffmpeg | 6.x | decoding/encoding |
| Node.js | 22 | yt-dlp's JavaScript runtime (`--js-runtimes node`), without it YouTube returns 360p-only / 403 |
| git | any | |
| NVIDIA driver (optional) | 560.94 (CUDA 12.6) | GPU separation: ~10x faster. CPU works, just slow (10–20 min/song) |
| cloudflared | 2026.8.x | the tunnel |

    sudo apt update && sudo apt install -y python3.12 python3.12-venv ffmpeg git nodejs
    # cloudflared (user-local, no sudo):
    mkdir -p ~/.local/bin && curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared

## 2. Code and Python environments

    git clone https://github.com/sashyo/drummate-chart.git ~/drum-notation
    cd ~/drum-notation
    ./run.sh   # creates .venv (CPU torch + demucs + audio-separator), then starts the app; Ctrl-C after it prints the URL

`run.sh` and `deploy/tunnel/start-chart.sh` prefer `.venv-cuda` when it exists. To build it (only with an NVIDIA card):

    python3 -m venv .venv-cuda && .venv-cuda/bin/pip install -U pip
    # pick the torch build that matches the DRIVER's CUDA version (nvidia-smi shows it).
    # driver 12.6 -> cu124; a cu130 build fails with "driver too old".
    .venv-cuda/bin/pip install "torch==2.6.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu124
    .venv-cuda/bin/pip install demucs "audio-separator[gpu]" -r requirements.txt
    .venv-cuda/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Notes for small cards: `backend/pipeline/gpu.py` disables cuDNN below 6 GB VRAM (a 3 GB GTX 1060
runs Demucs + DrumSep in ~1.2 GB with cuDNN off; with it on, cuDNN failed to initialise). Only
one GPU inference runs at a time (a lock in `gpu.py`); out-of-memory falls back to CPU per job.
`DRUMS_DEVICE=cpu` forces CPU. With a GPU present the fine-tuned Demucs (`htdemucs_ft`) is the
default separation model; `DRUMS_SEPARATION=htdemucs` keeps the plain one.

## 3. Restore state

    tar xzf ~/chart-data.tgz -C ~/drum-notation          # data/jobs, data/stats.json
    tar xzf ~/drumsep-model.tgz -C ~                     # ~/.cache/drumsep (optional)
    tar xzf ~/cloudflared-chart.tgz -C ~                 # ~/.cloudflared/{cert.pem, chart-config.yml, <tunnel-id>.json}
    chmod 600 ~/.cloudflared/*.json ~/.cloudflared/cert.pem

`chart-config.yml` must read (edit the home path if the user name differs):

    tunnel: <tunnel-id>
    credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json
    ingress:
      - hostname: chart.drummate.app
        service: http://localhost:8000
      - service: http_status:404

If the tunnel identity is NOT available (lost archive, or you want a fresh one):

    ~/.local/bin/cloudflared tunnel login            # opens a browser; pick the drummate.app zone
    ~/.local/bin/cloudflared tunnel create drummate-chart-2
    ~/.local/bin/cloudflared tunnel route dns --overwrite-dns drummate-chart-2 chart.drummate.app
    # then put the new tunnel id + credentials path into ~/.cloudflared/chart-config.yml

## 4. Start it

    cd ~/drum-notation && ./deploy/tunnel/start-chart.sh

That script is idempotent: starts the app on :8000 if `/api/health` isn't answering (using
`.venv-cuda` if present), then runs the tunnel. Logs: `/tmp/drummate-chart.log`, `/tmp/drummate-tunnel.log`.

**The public policy lives in `deploy/tunnel/env.public`** (upload-only, no YouTube, zero
retention). Every command that starts the live server must `. deploy/tunnel/env.public` first —
starting `uvicorn` by hand without it brings the site up on the personal-machine defaults
(links and YouTube on, audio kept). That happened once; check `/api/health` shows
`"links":false` and `"zeroRetention":true` after any restart.

Environment knobs (set in the shell that runs the script, or in the script):

| var | default | meaning |
|---|---|---|
| `DRUMS_WORKERS` | 2 | transcriptions in parallel (CPU threads are split between them) |
| `DRUMS_DEVICE` | auto | `cpu` to ignore the GPU |
| `DRUMS_SEPARATION` | auto | `htdemucs` to skip the fine-tuned model |
| `DRUMS_MAX_SECONDS` | 600 | cap on analysed audio per job |
| `DRUMS_DATA` | `<repo>/data` | where jobs/cache/stats live |
| `DRUMS_LINKS` | 1 (start-chart.sh sets 0) | accept links at all; the public site is upload-only |
| `DRUMS_ALLOW_YOUTUBE` | 1 (start-chart.sh sets 0) | fetch YouTube/streaming links; on by default for a personal machine, off on the public site |
| `DRUMS_YOUTUBE_WITH_CONSENT` | 1 (start-chart.sh sets 0) | with YouTube off, allow it behind a rights checkbox recorded on the job |
| `DRUMS_ZERO_RETENTION` | 0 (start-chart.sh sets 1) | the browser takes a finished chart's audio into memory and confirms; the server deletes its copies at once (or after `DRUMS_ZERO_RETENTION_GRACE_MIN`, 10, if nobody collects) |
| `DRUMS_SESSION_IDLE_MIN` | 20 | a chart's drum/backing tracks are released when its page closes, or after this idle time |
| `DRUMS_AUDIO_TTL_HOURS` | 6 | per-job drums/backing mp3s (what the chart plays) are deleted after this; charts/MIDI/MusicXML are kept |
| `DRUMS_KEEP_SOURCES` | 0 | `1` disables delete-on-use (sources, decoded wav and separation caches are otherwise removed the moment a job ends) — private builds only |
| `DRUMS_COOKIE_FILE` / `DRUMS_COOKIES_FROM_BROWSER` | unset | yt-dlp cookies, only relevant with `DRUMS_ALLOW_YOUTUBE=1` |

## 5. Verify (all four, in order)

    curl -s localhost:8000/api/health                      # {"ok":true,"demucs":true,"drumsep":true,...}
    curl -s https://chart.drummate.app/api/health          # same, through the tunnel
    curl -s https://chart.drummate.app/api/stats | head -c 300   # counter restored ("allTime" numbers non-zero)
    curl -s -X POST localhost:8000/api/upload -F "file=@some.wav" -F 'options={"renderAudio":true}'
    # (the public site is upload-only: any link answers 400; DRUMS_LINKS=1 / DRUMS_ALLOW_YOUTUBE=1 for a private build)
    # poll /api/jobs/<jobId> until "done" (~1 min for a 40 s clip on a GPU); then open the site and load it.
    # afterwards: the upload and its caches must be gone from data/cache (delete-on-use), the job dir keeps
    # score.json / drums.mid / drums.musicxml plus drums.mp3 / backing.mp3 for a few hours.

A finished chart from the old machine must also open: pick any id from `data/jobs/` and GET
`/api/jobs/<id>` — the server rehydrates finished jobs from disk.

## 6. Cut over

Both machines can run the tunnel with the same credentials for a moment; Cloudflare load-balances
between connectors, which is fine for a health check but not for jobs (a job lives in one
process). Sequence:

1. New machine: start-chart.sh, all four verifications pass.
2. Old machine: wait until `curl localhost:8000/api/queue` is `[]`, then stop the tunnel and the
   app (`pkill -f 'cloudflared tunnel' ; kill <uvicorn pid>`) and remove the Startup entry
   (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\DrumMateChart.cmd` on the WSL PC).
3. Copy `data/jobs` once more from old to new (charts made during the overlap), restart the new app.

Rollback is the reverse: start the old script again; the tunnel identity still works there.

## 7. Autostart after reboot

- **WSL2 on Windows**: `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\DrumMateChart.cmd` containing
  `wsl.exe -d Ubuntu -u <user> -- bash -lc /home/<user>/drum-notation/deploy/tunnel/start-chart.sh`
  (runs at logon; no admin needed. `schtasks` from inside WSL is denied.) A reboot wipes WSL's /tmp: keep
  nothing important there.
- **Native Linux**: a user systemd unit, e.g. `~/.config/systemd/user/drummate-chart.service` with
  `ExecStart=/home/<user>/drum-notation/deploy/tunnel/start-chart.sh`, `Type=forking`, then
  `systemctl --user enable --now drummate-chart` and `loginctl enable-linger <user>`.

## 8. Container path (a VPS instead of a PC)

`Dockerfile` + `docker-compose.yml` build a CPU-only image (`DRUMS_DATA=/data` volume). It works but
is slow (no GPU) and datacenter IPs trip YouTube's bot checks, so set `DRUMS_COOKIE_FILE`. The
tunnel runs the same way next to the container (point the ingress at the container's port).

## 9. Things that bit us, so you don't rediscover them

- **Two uvicorns on one port**: always check `ss -ltnp | grep :8000` before starting; the start
  script does. Restarting the app re-queues unfinished jobs by id, so users' pages keep working.
- **`pkill -f` with a pattern that appears in your own command line kills your shell** (and any log
  tail with that string). Match on `ss -ltnp` pids or `ps -eo pid,args | awk '$2 == ".venv-cuda/bin/python"'`.
- **Demucs `shifts` must be 0** (it is, in `separate.py`); the default random shift made results non-reproducible.
- **Non-Latin song titles** used to crash the MIDI export; fixed (titles are transliterated for MIDI meta text).
- **yt-dlp needs Node** for full-quality streams; without a JS runtime YouTube gives 360p only and 403s.
- **Cloudflare caches `app.js`/`style.css` at the edge regardless of the origin's `no-cache`** (it
  rewrites Cache-Control to `max-age=14400`), so after a deploy visitors ran old script. Assets are
  therefore versioned: `frontend/index.html` references `app.js?v=<build>`, and the build stamp
  must be bumped on every frontend change — `python tools/bump_build.py 2026-08-27a` updates all
  three places. The origin also sends `no-store` on the app shell.
- `frontend/notice.json` drives a site banner (edit the text, empty hides it; no restart).
