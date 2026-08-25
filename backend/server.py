"""HTTP API + static host for the drum transcription app."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline.run import ENGINE, Options, transcribe
from .pipeline.fetch import FetchError

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DRUMS_DATA", ROOT / "data"))
JOBS_DIR = DATA / "jobs"
CACHE_DIR = DATA / "cache"
FRONTEND = ROOT / "frontend"

MAX_SECONDS = float(os.environ.get("DRUMS_MAX_SECONDS", 600))

JOBS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DrumMate Chart")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Parallel jobs. Each job wants every core, so N workers each get 1/N of the
# CPU - same total throughput, but N users see their own job start and move
# instead of waiting behind a stranger's. Peak RAM is ~3-4 GB per job.
WORKERS = max(1, int(os.environ.get("DRUMS_WORKERS", "2")))
_pool = ThreadPoolExecutor(max_workers=WORKERS)
try:
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // WORKERS))
except Exception:  # noqa: BLE001
    pass
_lock = threading.Lock()
_pkg_locks: dict[str, threading.Lock] = {}


@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    message: str = "Queued"
    error: str | None = None
    title: str | None = None
    created: float = field(default_factory=time.time)
    score: dict | None = field(default=None, repr=False)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("score", None)
        d["hasScore"] = self.score is not None
        if self.status == "queued":
            queued_ahead = [j for j in _jobs.values()
                            if j.status == "queued" and j.created < self.created]
            running = [j for j in _jobs.values() if j.status == "running"]
            n = len(queued_ahead)
            if n == 0 and len(running) < WORKERS:
                d["message"] = "Queued \u2014 starting momentarily"
            else:
                d["message"] = (f"Queued: {n} ahead of you, {len(running)} running "
                                f"\u2014 {WORKERS} transcriptions run at a time")
        return d


_jobs: dict[str, Job] = {}
_futures: dict[str, object] = {}
_cancelled: set[str] = set()


class Cancelled(Exception):
    pass


class TranscribeRequest(BaseModel):
    url: str | None = None
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    beatsPerBar: int = Field(default=4, ge=2, le=12)
    tempo: float | None = Field(default=None, ge=30, le=300)
    sensitivity: float = Field(default=1.0, ge=0.3, le=2.5)
    maxSubdiv: int = Field(default=4)
    allowTriplets: bool = True
    detectToms: bool = True
    cymbalDetail: bool = True
    detectSwing: bool = True
    separation: str = "htdemucs"
    lockGrid: bool = False
    detector: str = "auto"
    renderAudio: bool = True


def _opts(req: TranscribeRequest) -> Options:
    start, end = req.start, req.end
    if end is not None and start is not None and end <= start:
        raise HTTPException(400, "End time must be after start time.")
    if end is None and start is not None:
        end = start + MAX_SECONDS
    if start is None and end is None:
        end = MAX_SECONDS
    return Options(
        start=start, end=end, beats_per_bar=req.beatsPerBar, fixed_tempo=req.tempo,
        sensitivity=req.sensitivity, max_subdiv=req.maxSubdiv,
        allow_triplets=req.allowTriplets, detect_toms=req.detectToms,
        cymbal_detail=req.cymbalDetail, detect_swing=req.detectSwing,
        separation=req.separation, render_audio=req.renderAudio,
        lock_grid=req.lockGrid, detector=req.detector,
        cookies_from_browser=os.environ.get("DRUMS_COOKIES_FROM_BROWSER"),
        cookie_file=os.environ.get("DRUMS_COOKIE_FILE"),
    )


def _run(job: Job, url: str | None, opts: Options, local: Path | None, title: str | None):
    out = JOBS_DIR / job.id

    def progress(p, msg):
        if job.id in _cancelled:
            raise Cancelled()
        with _lock:
            if p is not None:                      # None = message only (e.g. waiting for the GPU)
                job.progress = float(max(0.0, min(1.0, p)))
            job.message = str(msg)

    try:
        if job.id in _cancelled:
            raise Cancelled()
        with _lock:
            job.status, job.message = "running", "Starting"
            job.title = title or url
        doc = transcribe(url, out, CACHE_DIR, opts, progress=progress,
                         local_file=local, title=title)
        with _lock:
            job.score = doc
            job.title = doc.get("title")
            job.status, job.progress, job.message = "done", 1.0, "Done"
        _mark(job.id, "done")
        _count("done")
    except Cancelled:
        _mark(job.id, "error", error="Cancelled")
        with _lock:
            job.status, job.error, job.message = "error", "Cancelled", "Cancelled"
    except FetchError as exc:
        _mark(job.id, "error", error=str(exc))
        with _lock:
            job.status, job.error, job.message = "error", str(exc), "Failed"
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _mark(job.id, "error", error=f"{type(exc).__name__}: {exc}")
        with _lock:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "Failed"


def _mark(job_id: str, state: str, error: str | None = None) -> None:
    p = JOBS_DIR / job_id / "job.json"
    if p.exists():
        try:
            m = json.loads(p.read_text()); m["state"] = state
            if error:
                m["error"] = error
            p.write_text(json.dumps(m))
        except Exception:  # noqa: BLE001
            pass


@app.on_event("startup")
def _resume_unfinished():
    """Re-queue jobs that were running or queued when the server last stopped.

    Same ids, so a browser still polling them simply sees them finish. The
    download / separation caches make the re-run far cheaper than the first.
    """
    for p in sorted(JOBS_DIR.glob("*/job.json"), key=lambda x: x.stat().st_mtime):
        try:
            m = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if m.get("state") in ("done", "error") or (p.parent / "score.json").exists():
            continue
        if time.time() - m.get("created", 0) > 6 * 3600:
            continue                                   # stale; don't surprise anyone
        req = TranscribeRequest(**m.get("options", {}))
        local = Path(m["local"]) if m.get("local") else None
        if local and not local.exists():
            continue
        job = Job(id=m["id"], message="Re-queued after a server restart")
        job.sig = json.dumps(req.model_dump(), sort_keys=True)
        _jobs[job.id] = job
        _submit(job, m.get("url"), _opts(req), local, m.get("title"))
        print(f"resumed job {job.id}: {m.get('url') or m.get('title')}")


def _manifest(job_id: str, req: TranscribeRequest, url: str | None,
              local: Path | None, title: str | None) -> None:
    """Persist enough to re-queue this job if the server restarts mid-way."""
    d = JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps({
        "id": job_id, "url": url, "local": str(local) if local else None,
        "title": title, "options": req.model_dump(), "created": time.time()}))


def _submit(job: Job, url, opts, local, title):
    _futures[job.id] = _pool.submit(_run, job, url, opts, local, title)


@app.post("/api/transcribe")
def start(req: TranscribeRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(400, "Paste a link first.")
    url = req.url.strip()
    # the same link queued twice (double-click, refresh) should not wait
    # behind itself - hand back the pending job instead
    sig = json.dumps(req.model_dump(), sort_keys=True)
    for j in _jobs.values():
        if j.status in ("queued", "running") and getattr(j, "sig", None) == sig:
            return {"jobId": j.id, "duplicate": True}
    # ...and a link already charted with the same options (someone else's
    # Reddit find, or a refresh a day later) is handed back finished rather
    # than queued for another 15 minutes of separation
    done = _finished_with_sig(sig)
    if done:
        return {"jobId": done, "duplicate": True}
    job = Job(id=uuid.uuid4().hex[:12])
    job.sig = sig
    _jobs[job.id] = job
    _manifest(job.id, req, url, None, None)
    _submit(job, url, _opts(req), None, None)
    return {"jobId": job.id}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), options: str = Form("{}")):
    req = TranscribeRequest(**json.loads(options or "{}"))
    job = Job(id=uuid.uuid4().hex[:12])
    _jobs[job.id] = job
    dest = CACHE_DIR / f"upload_{job.id}_{Path(file.filename or 'audio').name}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    title = Path(file.filename or "Upload").stem
    _manifest(job.id, req, None, dest, title)
    _submit(job, None, _opts(req), dest, title)
    return {"jobId": job.id}


def _finished_with_sig(sig: str) -> str | None:
    for j in list(_jobs.values()):
        if j.status == "done" and getattr(j, "sig", None) == sig and j.score is not None:
            return j.id
    opts = json.loads(sig)
    best = None
    for p in JOBS_DIR.glob("*/job.json"):
        try:
            m = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if m.get("state") != "done" or not (p.parent / "score.json").exists():
            continue
        if m.get("options") != opts:
            continue
        try:
            eng = json.loads((p.parent / "score.json").read_text()).get("engine", 0)
        except Exception:  # noqa: BLE001
            continue
        if eng >= ENGINE and (best is None or m.get("created", 0) > best[0]):
            best = (m.get("created", 0), m["id"])
    return best[1] if best else None


def _get(job_id: str):
    """A job in memory, or a finished one rehydrated from disk. Finished
    jobs are not held across restarts, but their result page must keep
    working: a chart someone bookmarked is not allowed to 404 because the
    server was updated."""
    job = _jobs.get(job_id)
    if job is not None:
        return job
    d = JOBS_DIR / job_id
    if not (d / "score.json").exists():
        return None
    try:
        score = json.loads((d / "score.json").read_text())
        meta = json.loads((d / "job.json").read_text()) if (d / "job.json").exists() else {}
    except Exception:  # noqa: BLE001
        return None
    job = Job(id=job_id, status="done", progress=1.0, message="Done",
              title=score.get("title") or meta.get("title"),
              created=float(meta.get("created") or (d / "score.json").stat().st_mtime))
    job.score = score
    with _lock:
        _jobs.setdefault(job_id, job)
    return _jobs[job_id]


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    job = _get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return job.public()


@app.get("/api/jobs/{job_id}/score")
def get_score(job_id: str):
    job = _get(job_id)
    if job is None:
        path = JOBS_DIR / job_id / "score.json"
        if path.exists():
            return JSONResponse(json.loads(path.read_text()))
        raise HTTPException(404, "No such job")
    if job.score is None:
        raise HTTPException(409, "Not finished yet")
    return JSONResponse(job.score)


_ALLOWED = {"drums.mp3", "backing.mp3", "drums.mid", "drums.musicxml", "score.json"}


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str, dl: str | None = None):
    if name not in _ALLOWED:
        raise HTTPException(404, "Unknown file")
    path = (JOBS_DIR / job_id / name).resolve()
    if JOBS_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Not found")
    media = {"mp3": "audio/mpeg", "mid": "audio/midi",
             "musicxml": "application/vnd.recordare.musicxml+xml",
             "json": "application/json"}[name.rsplit(".", 1)[1]]
    # ?dl=<name> downloads under a song-named file ("Title - drumless.mp3")
    # instead of the internal backing.mp3; playback (no dl) streams as before
    if dl:
        safe = "".join(c for c in dl if c.isalnum() or c in " -_().")[:80].strip()
        return FileResponse(path, media_type=media, filename=safe or name)
    return FileResponse(path, media_type=media)


class ReexportRequest(BaseModel):
    bars: list[dict]


@app.post("/api/jobs/{job_id}/reexport")
def reexport(job_id: str, req: ReexportRequest):
    """Regenerate MIDI / MusicXML after the chart was edited in the browser."""
    from .pipeline.rebuild import reexport as _reexport

    job = _get(job_id)
    out = JOBS_DIR / job_id
    doc = job.score if job and job.score else None
    if doc is None:
        path = out / "score.json"
        if not path.exists():
            raise HTTPException(404, "No such job")
        doc = json.loads(path.read_text())
    new_doc = _reexport(doc, req.bars, out)
    if job is not None:
        job.score = new_doc
    return {"ok": True, "bars": len(new_doc["bars"])}


class RegridRequest(BaseModel):
    tempo: float = Field(ge=30, le=300)
    beatsPerBar: int | None = Field(default=None, ge=2, le=12)


@app.post("/api/jobs/{job_id}/regrid")
def regrid(job_id: str, req: RegridRequest):
    """Re-spell the chart at a different tempo / feel without re-analysing."""
    from .pipeline.rebuild import regrid as _regrid

    out = JOBS_DIR / job_id
    job = _get(job_id)
    doc = job.score if job and job.score else None
    if doc is None:
        path = out / "score.json"
        if not path.exists():
            raise HTTPException(404, "No such job")
        doc = json.loads(path.read_text())
    new_doc = _regrid(doc, req.tempo, out, req.beatsPerBar)
    if job is not None:
        job.score = new_doc
    return JSONResponse(new_doc)


@app.get("/api/jobs/{job_id}/clonehero")
def clonehero(job_id: str):
    """Build (or reuse) a Clone Hero song package for this transcription."""
    from .pipeline.clonehero import write_package
    from .pipeline.rebuild import from_json

    out = JOBS_DIR / job_id
    job = _get(job_id)
    doc = job.score if job and job.score else None
    if doc is None:
        path = out / "score.json"
        if not path.exists():
            raise HTTPException(404, "No such job")
        doc = json.loads(path.read_text())
    zpath = out / "clonehero.zip"
    score_json = out / "score.json"
    with _pkg_locks.setdefault(job_id, threading.Lock()):
        if not zpath.exists() or (score_json.exists()
                                  and score_json.stat().st_mtime > zpath.stat().st_mtime):
            write_package(doc, from_json(doc, []), out)
    safe = "".join(c for c in doc.get("title", "song") if c.isalnum() or c in " -_")[:60].strip() or "song"
    # read the whole file now: a rebuild (after an edit) replaces the zip
    # atomically, and streaming a path across that replacement produced
    # "Response content longer than Content-Length"
    from fastapi.responses import Response
    return Response(zpath.read_bytes(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{safe} [Clone Hero].zip"'})


@app.get("/api/queue")
def queue():
    """Everything queued or running, oldest first."""
    live = [j for j in _jobs.values() if j.status in ("queued", "running")]
    live.sort(key=lambda j: j.created)
    return [{"id": j.id, "status": j.status, "title": j.title, "message": j.public()["message"]}
            for j in live]


def _cancel(job_id: str) -> bool:
    job = _get(job_id)
    if job is None or job.status in ("done", "error"):
        return False
    _cancelled.add(job_id)
    fut = _futures.get(job_id)
    if fut is not None and fut.cancel():            # still queued: gone instantly
        _mark(job_id, "error")
        with _lock:
            job.status, job.error, job.message = "error", "Cancelled", "Cancelled"
    # otherwise it is running: it aborts at its next progress tick
    return True


@app.delete("/api/jobs/{job_id}")
def cancel(job_id: str):
    if not _cancel(job_id):
        raise HTTPException(404, "No such pending job")
    return {"ok": True}


@app.post("/api/queue/clear")
def clear_queue(keep: str | None = None):
    """Cancel every queued (not running) job, optionally keeping one."""
    n = 0
    for j in list(_jobs.values()):
        if j.status == "queued" and j.id != keep and _cancel(j.id):
            n += 1
    return {"cancelled": n}


@app.get("/api/health")
def health():
    from .pipeline.separate import demucs_available
    from .pipeline.drumsep import available as drumsep_available
    return {"ok": True, "demucs": demucs_available(), "drumsep": drumsep_available(),
            "maxSeconds": MAX_SECONDS, "workers": WORKERS}


# --------------------------------------------------------------------------
# Usage counter: unique visitors and submissions per day. IPs are hashed with
# a per-day salt and only the counts leave this file - nothing identifies a
# person, and the raw address is never written anywhere.
# --------------------------------------------------------------------------
STATS_PATH = DATA / "stats.json"
_stats_lock = threading.Lock()
_stats = {"days": {}, "all_visitors": [], "all_submitters": []}
try:
    _stats.update(json.loads(STATS_PATH.read_text()))
except Exception:  # noqa: BLE001
    pass
_stats_dirty = False


_active: dict[str, float] = {}          # hashed ip -> last seen (any app request)


def _touch(request) -> None:
    import hashlib
    ip = _client_ip(request)
    if ip in ("127.0.0.1", "::1", "?"):
        return
    h = hashlib.sha256(f"live|{ip}|drummate".encode()).hexdigest()[:16]
    now = time.time()
    with _stats_lock:
        _active[h] = now
        if len(_active) > 5000:
            for k in [k for k, t in _active.items() if now - t > 900]:
                _active.pop(k, None)


def _client_ip(request) -> str:
    return (request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "?"))


def _count(kind: str, request=None) -> None:
    global _stats_dirty
    import hashlib
    day = time.strftime("%Y-%m-%d")
    if kind == "done":
        with _stats_lock:
            d = _stats["days"].setdefault(day, {"visitors": [], "submitters": [], "submissions": 0, "charts_done": 0})
            d["charts_done"] += 1
            _stats_dirty = True
        return
    ip = _client_ip(request)
    if ip in ("127.0.0.1", "::1", "?"):
        return
    h = hashlib.sha256(f"{day}|{ip}|drummate".encode()).hexdigest()[:16]
    hall = hashlib.sha256(f"all|{ip}|drummate".encode()).hexdigest()[:16]
    with _stats_lock:
        d = _stats["days"].setdefault(day, {"visitors": [], "submitters": [], "submissions": 0, "charts_done": 0})
        if kind == "visit":
            if h not in d["visitors"]:
                d["visitors"].append(h)
            if hall not in _stats["all_visitors"]:
                _stats["all_visitors"].append(hall)
        elif kind == "submit":
            d["submissions"] += 1
            if h not in d["submitters"]:
                d["submitters"].append(h)
            if hall not in _stats["all_submitters"]:
                _stats["all_submitters"].append(hall)
        elif kind == "done":
            d["charts_done"] += 1
        _stats_dirty = True


def _flush_stats() -> None:
    global _stats_dirty
    while True:
        time.sleep(30)
        if _stats_dirty:
            with _stats_lock:
                try:
                    STATS_PATH.write_text(json.dumps(_stats))
                    _stats_dirty = False
                except Exception:  # noqa: BLE001
                    pass


threading.Thread(target=_flush_stats, daemon=True).start()


@app.get("/api/stats")
def stats():
    now = time.time()
    with _stats_lock:
        days = {day: {"visitors": len(d["visitors"]), "submitters": len(d["submitters"]),
                      "submissions": d["submissions"], "charts_done": d["charts_done"]}
                for day, d in sorted(_stats["days"].items())}
        active5 = sum(1 for t in _active.values() if now - t < 300)
        active15 = sum(1 for t in _active.values() if now - t < 900)
        queue = [{"id": j.id, "status": j.status, "title": (j.title or "")[:60]}
                 for j in _jobs.values() if j.status in ("queued", "running")]
        return {"now": {"active5min": active5, "active15min": active15,
                        "running": sum(1 for j in queue if j["status"] == "running"),
                        "queued": sum(1 for j in queue if j["status"] == "queued"), "queue": queue},
                "days": days, "allTime": {"visitors": len(_stats["all_visitors"]),
                                          "submitters": len(_stats["all_submitters"]),
                                          "submissions": sum(d["submissions"] for d in _stats["days"].values()),
                                          "charts_done": sum(d["charts_done"] for d in _stats["days"].values())}}


@app.middleware("http")
async def _revalidate_app_shell(request, call_next):
    """The app shell must revalidate on every load (ETag makes it a cheap 304),
    otherwise browsers heuristically cache app.js and users run stale code."""
    if request.method == "GET" and request.url.path == "/":
        _count("visit", request)
    elif request.method == "POST" and request.url.path == "/api/transcribe":
        _count("submit", request)
    if request.url.path == "/" or request.url.path.startswith("/api/jobs/"):
        _touch(request)
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".js", ".css", ".html", ".svg")):
        resp.headers["Cache-Control"] = "no-cache"
    elif path.startswith("/api/jobs/") and path.endswith((".mp3", ".ogg")):
        resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="static")
