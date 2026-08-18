"""Stage 6 - turn quantised hits into readable drum notation.

Notation is built beat by beat. Within a beat every hit's written duration
runs to the next hit in the same voice, and unplayed slots become rests, so
beams group per beat the way a human copyist would write them. Two voices
are used, as is standard for drumset: stems up for hands (cymbals, snare,
toms), stems down for feet (bass drum, hi-hat pedal).
"""
from __future__ import annotations

from . import drums as K
from .quantize import PPQ, QScore

# slots-in-beat -> (vexflow duration, dots), per subdivision of the beat
_TABLES: dict[int, dict[int, tuple[str, int]]] = {
    1: {1: ("q", 0)},
    2: {1: ("8", 0), 2: ("q", 0)},
    4: {1: ("16", 0), 2: ("8", 0), 3: ("8", 1), 4: ("q", 0)},
    8: {1: ("32", 0), 2: ("16", 0), 3: ("16", 1), 4: ("8", 0), 6: ("8", 1), 8: ("q", 0)},
    3: {1: ("8", 0), 2: ("q", 0), 3: ("q", 1)},          # inside a 3:2 tuplet
    6: {1: ("16", 0), 2: ("8", 0), 3: ("8", 1), 4: ("q", 0), 6: ("q", 1)},  # 6:4
}
_TUPLET_OCCUPIED = {3: 2, 6: 4}


def _split(count: int, table: dict[int, tuple[str, int]],
           rest: bool = False) -> list[tuple[str, int, int]]:
    """Express `count` slots as (duration, dots, slots) pieces, largest first.

    Rests avoid dotted values - an 8th + 16th rest reads at a glance where a
    dotted-8th rest makes the eye stop and count.
    """
    out = []
    remaining = count
    sizes = sorted((k for k in table
                    if not (rest and table[k][1])), reverse=True)
    guard = 0
    while remaining > 0 and guard < 16:
        guard += 1
        pick = next((s for s in sizes if s <= remaining), None)
        if pick is None:
            break
        dur, dots = table[pick]
        out.append((dur, dots, pick))
        remaining -= pick
    return out


def build(q: QScore, meta: dict) -> dict:
    bars_json = []
    for bar in q.bars:
        voices = {K.UP: [], K.DOWN: []}
        for vname in (K.UP, K.DOWN):
            notes = [n for n in bar.notes if K.voice_of(n.inst) == vname]
            voices[vname] = _build_voice(notes, bar, q)
        # Charts don't stack rests under the hands: whole-beat rests in the
        # feet voice keep the timing math but are not printed.
        for e in voices[K.DOWN]:
            if e["type"] == "rest" and e["dur"] == "q" and not e["dots"]:
                e["hidden"] = True
        bars_json.append({
            "index": bar.index,
            "number": bar.index + 1,
            "startTime": round(bar.start_time, 4),
            "endTime": round(bar.end_time, 4),
            "grid": bar.grid_name,
            "subdivision": bar.subdivision,
            "empty": len(bar.notes) == 0,
            "voices": voices,
            # Raw hits are the editable source of truth; the browser re-runs
            # the same layout on them after you change something.
            "hits": [{
                "tick": n.tick, "inst": n.inst,
                "velocity": round(n.velocity, 3),
                "ghost": n.ghost, "accent": n.accent, "flam": n.flam,
                "time": round(n.time, 4),
            } for n in sorted(bar.notes, key=lambda x: (x.tick, x.inst))],
        })

    return {
        "title": meta.get("title", "Untitled"),
        "source": meta.get("source"),
        "videoId": meta.get("videoId"),
        "tempo": round(q.tempo, 1),
        "timeSignature": f"{q.beats_per_bar}/4",
        "beatsPerBar": q.beats_per_bar,
        "swing": q.swing,
        "swingRatio": round(q.swing_ratio, 3),
        "ppq": q.ppq,
        "ticksPerBar": q.ticks_per_bar,
        "bars": bars_json,
        "kit": sorted({n.inst for bar in q.bars for n in bar.notes},
                      key=lambda i: K.DRUMS.get(i, {}).get("order", 99)),
        "stats": meta.get("stats", {}),
        "separation": meta.get("separation"),
        "audio": meta.get("audio", {}),
        "offset": round(meta.get("offset", 0.0), 4),
    }


def _build_voice(notes, bar, q: QScore) -> list[dict]:
    n = bar.subdivision
    table = _TABLES.get(n, _TABLES[4])
    slot_ticks = PPQ // n
    elems: list[dict] = []

    for beat in range(q.beats_per_bar):
        lo, hi = beat * PPQ, (beat + 1) * PPQ
        in_beat = [x for x in notes if lo <= x.tick < hi]
        slots: dict[int, list] = {}
        for x in in_beat:
            s = min(n - 1, (x.tick - lo) // slot_ticks)
            slots.setdefault(int(s), []).append(x)

        beat_elems: list[dict] = []
        if not slots:
            beat_elems.append(_rest("q", 0, beat, 0, bar, q))
        else:
            occupied = sorted(slots)
            if occupied[0] > 0:
                for dur, dots, used in _split(occupied[0], table, rest=True):
                    beat_elems.append(_rest(dur, dots, beat, 0, bar, q))
            for i, s in enumerate(occupied):
                nxt = occupied[i + 1] if i + 1 < len(occupied) else n
                count = nxt - s
                pieces = _split(count, table)
                if not pieces:
                    pieces = [(table[min(table)][0], table[min(table)][1], 1)]
                dur, dots, used = pieces[0]
                beat_elems.append(_note(slots[s], dur, dots, beat, s, bar, q))
                for dur2, dots2, _u in pieces[1:]:
                    beat_elems.append(_rest(dur2, dots2, beat, s, bar, q))

        if n in _TUPLET_OCCUPIED:
            if len(beat_elems) > 1:
                for e in beat_elems:
                    e["tuplet"] = {"num": n, "den": _TUPLET_OCCUPIED[n]}
            else:
                # One event filling a whole triplet beat is just a quarter
                # note - writing it as a dotted quarter inside no tuplet
                # would be half a beat too long.
                beat_elems[0]["dur"] = "q"
                beat_elems[0]["dots"] = 0
        elems.extend(beat_elems)
    return elems


def _note(hits, dur, dots, beat, slot, bar, q) -> dict:
    hits = sorted(hits, key=lambda h: K.DRUMS.get(h.inst, {}).get("order", 99))
    return {
        "type": "note",
        "dur": dur,
        "dots": dots,
        "beat": beat,
        "keys": [K.key_of(h.inst) for h in hits],
        "insts": [h.inst for h in hits],
        "accent": any(h.accent for h in hits),
        "ghost": all(h.ghost for h in hits) and any(h.inst == "snare" for h in hits),
        "flam": any(h.flam for h in hits),
        "open": any(h.inst == "openhh" for h in hits),
        "velocity": round(max(h.velocity for h in hits), 3),
        "time": round(min(h.time for h in hits), 4),
        "tuplet": None,
    }


def _rest(dur, dots, beat, slot, bar, q) -> dict:
    t = bar.start_time + (beat + slot / max(bar.subdivision, 1)) * (
        (bar.end_time - bar.start_time) / max(q.beats_per_bar, 1))
    return {
        "type": "rest", "dur": dur, "dots": dots, "beat": beat,
        "keys": [], "insts": [], "accent": False, "ghost": False,
        "flam": False, "open": False, "velocity": 0.0,
        "time": round(t, 4), "tuplet": None,
    }
