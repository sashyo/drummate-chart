"""Stage 1 - get audio onto disk.

Accepts a YouTube (or any yt-dlp supported) URL, or a locally uploaded file.
Downloads the best audio-only stream and normalises it to 44.1 kHz WAV so
every later stage can assume the same format.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

SR = 44100


@dataclass
class Source:
    path: Path          # normalised wav (stereo, 44.1k)
    title: str
    duration: float
    video_id: str | None
    webpage_url: str | None


def _slug(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _to_wav(src: Path, dst: Path, start: float | None, end: float | None) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if end:
        dur = end - (start or 0.0)
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def from_url(
    url: str,
    cache_dir: Path,
    progress=None,
    start: float | None = None,
    end: float | None = None,
    cookies_from_browser: str | None = None,
    cookie_file: str | None = None,
) -> Source:
    """Download audio for `url` into `cache_dir` and return a normalised wav."""
    import yt_dlp

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _slug(url)
    meta_path = cache_dir / f"{key}.json"
    wav_path = cache_dir / f"{key}_{int((start or 0)*1000)}_{int((end or 0)*1000)}.wav"

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                progress(0.02 + 0.08 * done / total, "Downloading audio")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(cache_dir / f"{key}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "progress_hooks": [hook],
        "retries": 3,
        "concurrent_fragment_downloads": 4,
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookie_file:
        opts["cookiefile"] = cookie_file

    # Reuse a previous download when we already have one for this URL.
    existing = [
        p for p in cache_dir.glob(f"{key}.*")
        if p.suffix.lower() not in (".json", ".wav", ".part")
    ]
    if existing and meta_path.exists():
        info = json.loads(meta_path.read_text())
        media = existing[0]
    else:
        info, ydl = _download(url, opts)
        media = Path(ydl.prepare_filename(info))
        if not media.exists():
            cands = [
                p for p in cache_dir.glob(f"{key}.*")
                if p.suffix.lower() not in (".json", ".wav", ".part")
            ]
            if not cands:
                raise FetchError("Download finished but no audio file was produced.")
            media = cands[0]
        meta_path.write_text(json.dumps({
            "title": info.get("title") or url,
            "duration": info.get("duration") or 0,
            "id": info.get("id"),
            "webpage_url": info.get("webpage_url") or url,
        }))
        info = json.loads(meta_path.read_text())

    if progress:
        progress(0.10, "Decoding audio")
    # Same media, same clip -> same wav; don't re-decode (keeps the stem
    # caches, keyed on content, warm across repeats of a song).
    if not (wav_path.exists() and wav_path.stat().st_size > 1000
            and wav_path.stat().st_mtime >= media.stat().st_mtime):
        _to_wav(media, wav_path, start, end)

    from . import scratch
    scratch.note(media); scratch.note(wav_path)
    return Source(
        path=wav_path,
        title=info.get("title") or url,
        duration=float(info.get("duration") or 0.0),
        video_id=info.get("id"),
        webpage_url=info.get("webpage_url") or url,
    )


# YouTube rejects some player clients outright (HTTP 403 / "format is not
# available"), and which ones work changes over time. Try them in order.
YT_CLIENTS = [c for c in os.environ.get(
    "DRUMS_YT_CLIENTS", "android,tv,web_safari,ios,web").split(",") if c.strip()]


def _download(url: str, opts: dict):
    """Download `url`, falling back through player clients on failure."""
    import yt_dlp

    last: Exception | None = None
    attempts = [{"youtube": {"player_client": [c]}} for c in YT_CLIENTS] or [None]
    for extractor_args in attempts:
        trial = dict(opts)
        if extractor_args:
            trial["extractor_args"] = extractor_args
        try:
            ydl = yt_dlp.YoutubeDL(trial)
            with ydl:
                return ydl.extract_info(url, download=True), ydl
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc)
            # A genuinely missing / private video will not be fixed by
            # trying another client, so stop early.
            if any(k in msg for k in ("Private video", "Video unavailable",
                                      "removed by the uploader",
                                      "members-only", "Sign in to confirm")):
                break
    raise FetchError(_friendly(last or RuntimeError("download failed")))


def from_file(
    src: Path,
    cache_dir: Path,
    title: str,
    start: float | None = None,
    end: float | None = None,
) -> Source:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _slug(str(src) + title)
    wav_path = cache_dir / f"{key}_{int((start or 0)*1000)}_{int((end or 0)*1000)}.wav"
    _to_wav(src, wav_path, start, end)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(wav_path)],
        capture_output=True, text=True, check=True).stdout.strip() or 0.0)
    from . import scratch
    scratch.note(wav_path)
    return Source(path=wav_path, title=title, duration=dur, video_id=None, webpage_url=None)


class FetchError(RuntimeError):
    pass


def _friendly(exc: Exception) -> str:
    msg = str(exc)
    if "Sign in to confirm" in msg or "bot" in msg.lower():
        return ("YouTube asked this server to prove it isn't a bot. Set "
                "DRUMS_COOKIES_FROM_BROWSER=chrome (or firefox) or point "
                "DRUMS_COOKIE_FILE at a cookies.txt export, then retry.")
    if "Private video" in msg:
        return "That video is private."
    if "Video unavailable" in msg:
        return "That video is unavailable (removed, or blocked in this region)."
    if "age" in msg.lower() and "restrict" in msg.lower():
        return "That video is age-restricted; a cookies file is required."
    m = re.search(r"ERROR:\s*(.+)", msg)
    return m.group(1) if m else f"Could not download that link: {msg[:300]}"
