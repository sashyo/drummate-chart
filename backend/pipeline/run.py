"""The full transcription pipeline, start to finish."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import exports, fetch, onsets, quantize, rhythm, score, separate


@dataclass
class Options:
    start: float | None = None
    end: float | None = None
    beats_per_bar: int = 4
    fixed_tempo: float | None = None
    lock_grid: bool = False
    sensitivity: float = 1.0
    max_subdiv: int = 4
    allow_triplets: bool = True
    detect_toms: bool = True
    cymbal_detail: bool = True
    detect_swing: bool = True
    separation: str = "htdemucs"      # or "none" to skip / "hpss"
    render_audio: bool = True
    cookies_from_browser: str | None = None
    cookie_file: str | None = None


def transcribe(url: str | None, out_dir: Path, cache_dir: Path,
               opts: Options, progress=None, local_file: Path | None = None,
               title: str | None = None) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    def note(p, msg):
        if progress:
            progress(p, msg)

    note(0.01, "Fetching audio")
    if local_file is not None:
        src = fetch.from_file(local_file, cache_dir, title or local_file.stem,
                              opts.start, opts.end)
    else:
        src = fetch.from_url(url, cache_dir, progress=progress, start=opts.start,
                             end=opts.end,
                             cookies_from_browser=opts.cookies_from_browser,
                             cookie_file=opts.cookie_file)

    import librosa
    import numpy as np

    mix, _ = librosa.load(str(src.path), sr=separate.SR, mono=True)

    if opts.separation == "none":
        note(0.30, "Skipping separation")
        stem = separate.Stems(mono=mix.astype(np.float32), method="none")
    else:
        model = opts.separation if opts.separation.startswith("htdemucs") else "htdemucs"
        stem = separate.drum_stem(src.path, out_dir, cache_dir, progress=progress,
                                  model=model, render_audio=opts.render_audio)
        if opts.separation == "hpss":
            stem = separate.Stems(mono=separate._hpss_fallback(src.path, progress)[0].mean(0),
                                  method="hpss")

    n = min(len(stem.mono), len(mix))
    grid = rhythm.analyse(stem.mono[:n], mix=mix[:n], progress=progress,
                          beats_per_bar=opts.beats_per_bar,
                          fixed_tempo=opts.fixed_tempo,
                          lock_grid=opts.lock_grid)

    det = onsets.detect(stem.mono, progress=progress, sensitivity=opts.sensitivity,
                        detect_toms=opts.detect_toms, cymbal_detail=opts.cymbal_detail)

    q = quantize.quantize(det.hits, grid, progress=progress,
                          max_subdiv=opts.max_subdiv,
                          allow_triplets=opts.allow_triplets,
                          detect_swing=opts.detect_swing)

    audio_files = {}
    for name, p in (stem.files or {}).items():
        audio_files[name] = p.name

    meta = {
        "title": src.title,
        "source": src.webpage_url,
        "videoId": src.video_id,
        "separation": stem.method,
        "offset": opts.start or 0.0,
        "audio": audio_files,
        "stats": {
            "hits": len(det.hits),
            "counts": det.debug.get("counts", {}),
            "bars": len(q.bars),
            "duration": round(det.duration, 2),
            "analysisSeconds": round(time.time() - t0, 1),
        },
    }

    note(0.90, "Engraving the chart")
    doc = score.build(q, meta)

    note(0.94, "Writing MIDI and MusicXML")
    exports.write_midi(doc, q, out_dir / "drums.mid")
    exports.write_musicxml(doc, out_dir / "drums.musicxml")
    (out_dir / "score.json").write_text(json.dumps(doc, indent=1))

    doc["downloads"] = {
        "midi": "drums.mid",
        "musicxml": "drums.musicxml",
        "json": "score.json",
    }
    note(1.0, "Done")
    return doc
