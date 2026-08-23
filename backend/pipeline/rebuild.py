"""Re-engrave and re-export a chart after the user has edited it in the browser.

The browser owns the edited hit list; this turns that back into the same
QScore the analysis pipeline produces, so exports stay identical in shape.
"""
from __future__ import annotations

from pathlib import Path

from . import exports, score
from .quantize import PPQ, QBar, QNote, QScore


def from_json(doc: dict, edited_bars: list[dict]) -> QScore:
    by_index = {int(b["index"]): b for b in edited_bars}
    bars: list[QBar] = []
    for b in doc["bars"]:
        idx = int(b["index"])
        hits = by_index.get(idx, b).get("hits", [])
        bar = QBar(
            index=idx,
            subdivision=int(b.get("subdivision", 4)),
            grid_name=b.get("grid", "16th"),
            start_time=float(b.get("startTime", 0.0)),
            end_time=float(b.get("endTime", 0.0)),
        )
        for h in hits:
            bar.notes.append(QNote(
                tick=int(h["tick"]), inst=str(h["inst"]),
                velocity=float(h.get("velocity", 0.8)),
                ghost=bool(h.get("ghost")), accent=bool(h.get("accent")),
                flam=bool(h.get("flam")), time=float(h.get("time", bar.start_time)),
            ))
        bar.notes.sort(key=lambda n: (n.tick, n.inst))
        bars.append(bar)

    return QScore(
        bars=bars, ppq=PPQ,
        beats_per_bar=int(doc.get("beatsPerBar", 4)),
        ticks_per_bar=int(doc.get("ticksPerBar", PPQ * 4)),
        swing=bool(doc.get("swing")), swing_ratio=float(doc.get("swingRatio", 0.5)),
        tempo=float(doc.get("tempo", 120.0)),
    )


def reexport(doc: dict, edited_bars: list[dict], out_dir: Path) -> dict:
    q = from_json(doc, edited_bars)
    meta = {
        "title": doc.get("title", "Untitled"),
        "source": doc.get("source"),
        "videoId": doc.get("videoId"),
        "separation": doc.get("separation"),
        "offset": doc.get("offset", 0.0),
        "audio": doc.get("audio", {}),
        "stats": doc.get("stats", {}),
    }
    new_doc = score.build(q, meta)
    new_doc["downloads"] = doc.get("downloads", {})
    exports.write_midi(new_doc, q, out_dir / "drums.mid")
    exports.write_musicxml(new_doc, out_dir / "drums.musicxml")
    import json
    (out_dir / "score.json").write_text(json.dumps(new_doc, indent=1))
    return new_doc


def regrid(doc: dict, tempo: float, out_dir: Path, beats_per_bar: int | None = None) -> dict:
    """Re-spell the same hits on a grid locked to `tempo` (e.g. half-time).

    Hits keep their detected times; only the grid, bar lines and notation
    change. Seconds, not minutes - no audio is touched.
    """
    import json
    from .onsets import Hit
    from . import rhythm, quantize, score as SC, exports

    bpb = int(beats_per_bar or doc.get("beatsPerBar", 4))
    hits = [Hit(time=float(h["time"]), inst=h["inst"], velocity=float(h.get("velocity", 0.8)),
                ghost=bool(h.get("ghost")), accent=bool(h.get("accent")))
            for b in doc["bars"] for h in b["hits"]]
    hits.sort(key=lambda h: h.time)
    duration = float(doc.get("stats", {}).get("duration", hits[-1].time + 2 if hits else 10))
    grid = rhythm.grid_from_drums(hits, duration, bpb, fixed_tempo=float(tempo))
    if grid is None:
        raise ValueError("not enough hits to build a grid")
    grid = rhythm.refine_with_hits(grid, hits)
    q = quantize.quantize(hits, grid)
    meta = {k: doc.get(k) for k in ("title", "source", "videoId", "separation", "detector", "engine", "offset", "audio")}
    meta["stats"] = dict(doc.get("stats", {}), bars=len(q.bars))
    new_doc = SC.build(q, meta)
    new_doc["downloads"] = doc.get("downloads", {})
    exports.write_midi(new_doc, q, out_dir / "drums.mid")
    exports.write_musicxml(new_doc, out_dir / "drums.musicxml")
    (out_dir / "score.json").write_text(json.dumps(new_doc, indent=1))
    return new_doc
