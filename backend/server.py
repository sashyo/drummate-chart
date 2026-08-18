"""HTTP API + static host for the drum transcription app."""
from __future__ import annotations

import json
import os
import shutil
import threading
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

from .pipeline.run import Options, transcribe
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

_pool = ThreadPoolExecutor(max_workers=1)   # separation is CPU-hungry; one at a time
_lock = threading.Lock()


@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    message: str = "Queued"
    error: str | None = None
    title: str | None = None
    score: dict | None = field(default=None, repr=False)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("score", None)
        d["hasScore"] = self.score is not None
        return d


_jobs: dict[str, Job] = {}


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
        cookies_from_browser=os.environ.get("DRUMS_COOKIES_FROM_BROWSER"),
        cookie_file=os.environ.get("DRUMS_COOKIE_FILE"),
    )


def _run(job: Job, url: str | None, opts: Options, local: Path | None, title: str | None):
    out = JOBS_DIR / job.id

    def progress(p, msg):
        with _lock:
            job.progress = float(max(0.0, min(1.0, p)))
            job.message = str(msg)

    try:
        with _lock:
            job.status, job.message = "running", "Starting"
        doc = transcribe(url, out, CACHE_DIR, opts, progress=progress,
                         local_file=local, title=title)
        with _lock:
            job.score = doc
            job.title = doc.get("title")
            job.status, job.progress, job.message = "done", 1.0, "Done"
    except FetchError as exc:
        with _lock:
            job.status, job.error, job.message = "error", str(exc), "Failed"
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        with _lock:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "Failed"


@app.post("/api/transcribe")
def start(req: TranscribeRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(400, "Paste a link first.")
    job = Job(id=uuid.uuid4().hex[:12])
    _jobs[job.id] = job
    _pool.submit(_run, job, req.url.strip(), _opts(req), None, None)
    return {"jobId": job.id}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), options: str = Form("{}")):
    req = TranscribeRequest(**json.loads(options or "{}"))
    job = Job(id=uuid.uuid4().hex[:12])
    _jobs[job.id] = job
    dest = CACHE_DIR / f"upload_{job.id}_{Path(file.filename or 'audio').name}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    _pool.submit(_run, job, None, _opts(req), dest, Path(file.filename or "Upload").stem)
    return {"jobId": job.id}


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return job.public()


@app.get("/api/jobs/{job_id}/score")
def get_score(job_id: str):
    job = _jobs.get(job_id)
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
def get_file(job_id: str, name: str):
    if name not in _ALLOWED:
        raise HTTPException(404, "Unknown file")
    path = (JOBS_DIR / job_id / name).resolve()
    if JOBS_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Not found")
    media = {"mp3": "audio/mpeg", "mid": "audio/midi",
             "musicxml": "application/vnd.recordare.musicxml+xml",
             "json": "application/json"}[name.rsplit(".", 1)[1]]
    return FileResponse(path, media_type=media, filename=name)


class ReexportRequest(BaseModel):
    bars: list[dict]


@app.post("/api/jobs/{job_id}/reexport")
def reexport(job_id: str, req: ReexportRequest):
    """Regenerate MIDI / MusicXML after the chart was edited in the browser."""
    from .pipeline.rebuild import reexport as _reexport

    job = _jobs.get(job_id)
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


@app.get("/api/health")
def health():
    from .pipeline.separate import demucs_available
    return {"ok": True, "demucs": demucs_available(), "maxSeconds": MAX_SECONDS}


app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="static")
