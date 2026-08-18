"""Stage 5 - snap raw hit times onto a musical grid.

Works in beat space (from BeatGrid) so tempo drift doesn't smear the grid.
Each bar picks its own subdivision - straight 8ths/16ths/32nds or triplets -
by whichever snaps the bar's hits with least error, with a penalty against
needlessly fine grids. Swing is detected globally and, when present, the
notation reverts to straight 8ths plus a "Swing" instruction, which is how
drum charts are actually written.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PPQ = 48                       # ticks per quarter note (divisible by 16ths and triplets)

# subdivisions per beat -> (name, is_triplet, complexity penalty)
GRIDS = {
    1: ("quarter", False, 0.0),
    2: ("8th", False, 0.004),
    4: ("16th", False, 0.010),
    8: ("32nd", False, 0.030),
    3: ("8th-triplet", True, 0.014),
    6: ("16th-triplet", True, 0.034),
}


@dataclass
class QNote:
    tick: int                  # 0 .. ticks_per_bar-1
    inst: str
    velocity: float
    ghost: bool = False
    accent: bool = False
    flam: bool = False
    time: float = 0.0          # original seconds, kept for playback sync


@dataclass
class QBar:
    index: int                 # 0-based, may be -1 for a pickup bar
    notes: list[QNote] = field(default_factory=list)
    subdivision: int = 4
    grid_name: str = "16th"
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class QScore:
    bars: list[QBar]
    ppq: int
    beats_per_bar: int
    ticks_per_bar: int
    swing: bool
    swing_ratio: float
    tempo: float


def quantize(hits, grid, progress=None, max_subdiv: int = 4,
             allow_triplets: bool = True, detect_swing: bool = True) -> QScore:
    if progress:
        progress(0.82, "Quantising to a rhythmic grid")

    bpb = grid.beats_per_bar
    ticks_per_bar = PPQ * bpb

    times = np.array([h.time for h in hits], dtype=float)
    if len(times) == 0:
        return QScore([], PPQ, bpb, ticks_per_bar, False, 0.5, grid.tempo)

    beats = np.asarray(grid.to_beats(times), dtype=float)
    rel = beats - grid.bar_zero_beat            # beats since the first downbeat
    bar_idx = np.floor(rel / bpb).astype(int)
    pos = rel - bar_idx * bpb                   # 0..bpb within the bar

    swing, ratio = (_detect_swing(hits, pos) if detect_swing else (False, 0.5))

    allowed = [n for n in (1, 2, 4, 8) if n <= max_subdiv]
    if allow_triplets and not swing:
        allowed += [n for n in (3, 6) if n <= max_subdiv * 2]
    if not allowed:
        allowed = [4]

    bars: dict[int, QBar] = {}
    for b in sorted(set(bar_idx.tolist())):
        sel = np.where(bar_idx == b)[0]
        p = pos[sel]
        best_n, best_cost = allowed[0], float("inf")
        for n in allowed:
            step = 1.0 / n
            err = np.abs(p - np.round(p / step) * step)
            cost = float(err.mean()) + GRIDS[n][2]
            if cost < best_cost:
                best_n, best_cost = n, cost
        bar = QBar(index=int(b), subdivision=best_n, grid_name=GRIDS[best_n][0])

        step = 1.0 / best_n
        for i in sel:
            snapped_beat = float(np.round(pos[i] / step) * step)
            tick = int(round(snapped_beat * PPQ))
            tick = max(0, min(ticks_per_bar, tick))
            h = hits[i]
            bar.notes.append(QNote(
                tick=tick, inst=h.inst, velocity=float(h.velocity),
                ghost=h.ghost, accent=h.accent, time=float(h.time)))
        bars[int(b)] = bar

    # A note snapped to the very end of a bar belongs to the next downbeat.
    for b in sorted(bars):
        carry = [n for n in bars[b].notes if n.tick >= ticks_per_bar]
        if carry:
            bars[b].notes = [n for n in bars[b].notes if n.tick < ticks_per_bar]
            nxt = bars.setdefault(b + 1, QBar(index=b + 1, subdivision=bars[b].subdivision,
                                              grid_name=bars[b].grid_name))
            for n in carry:
                n.tick = 0
                nxt.notes.append(n)

    ordered = [bars[b] for b in sorted(bars)]
    for bar in ordered:
        bar.notes = _merge(bar.notes)
        bar.notes.sort(key=lambda n: (n.tick, n.inst))
        bar.start_time = float(grid.to_time(grid.bar_zero_beat + bar.index * bpb))
        bar.end_time = float(grid.to_time(grid.bar_zero_beat + (bar.index + 1) * bpb))

    _cleanup(ordered, bpb)
    ordered = _fill_empty_bars(ordered, grid, bpb)
    ordered = _trim_edges(ordered)

    return QScore(bars=ordered, ppq=PPQ, beats_per_bar=bpb, ticks_per_bar=ticks_per_bar,
                  swing=swing, swing_ratio=ratio, tempo=grid.tempo)


def _merge(notes: list[QNote]) -> list[QNote]:
    """Two hits of one drum on one tick = one note (a very close pair = a flam)."""
    out: dict[tuple[int, str], QNote] = {}
    for n in sorted(notes, key=lambda x: x.time):
        key = (n.tick, n.inst)
        prev = out.get(key)
        if prev is None:
            out[key] = n
            continue
        gap = abs(n.time - prev.time)
        # Only a deliberate double stroke reads as a flam - detection jitter
        # produces the same timing with one hit far quieter.
        if 0.012 < gap < 0.075 and min(n.velocity, prev.velocity) >= 0.35:
            prev.flam = True
        prev.velocity = max(prev.velocity, n.velocity)
        prev.accent = prev.accent or n.accent
        prev.ghost = prev.ghost and n.ghost
    return list(out.values())


CYMS = {"hihat", "openhh", "ride"}


def _cleanup(bars: list[QBar], bpb: int) -> None:
    """Remove detection jitter so the same groove spells the same way.

    One repair, deliberately conservative: when the cymbal line fills >=80%%
    of a bar's grid, a lone one-slot hole inside the run is a detection miss -
    a drummer does not skip one 16th mid-run. Genuinely sparse patterns
    (offbeat hats fill 50%%) never reach the threshold and are left alone.
    (An earlier "echo removal" rule was measured to eat real alternating-
    velocity 16ths and was dropped; same-tick doubles are merged upstream.)
    """
    for bar in bars:
        n = bar.subdivision
        slot = PPQ // n
        total = n * bpb
        cym = [x for x in bar.notes if x.inst in CYMS]
        have = {x.tick // slot for x in cym}
        if total and len(have) / total >= 0.8:
            vel = float(np.median([x.velocity for x in cym]))
            span = bar.end_time - bar.start_time
            for sl in range(1, total - 1):
                if sl in have or (sl - 1) not in have or (sl + 1) not in have:
                    continue
                bar.notes.append(QNote(
                    tick=sl * slot, inst="hihat", velocity=vel,
                    time=bar.start_time + span * sl / total))
            bar.notes.sort(key=lambda x: (x.tick, x.inst))


def _trim_edges(bars: list[QBar]) -> list[QBar]:
    """Drop silent bars at the very start and end.

    Beat tracking often opens a bar before the drums actually enter, which
    would otherwise print as a blank pickup measure at the head of the chart.
    """
    lo, hi = 0, len(bars) - 1
    while lo <= hi and not bars[lo].notes:
        lo += 1
    while hi >= lo and not bars[hi].notes:
        hi -= 1
    return bars[lo:hi + 1]


def _fill_empty_bars(bars: list[QBar], grid, bpb: int) -> list[QBar]:
    """Insert genuinely empty bars so the chart keeps its bar numbering."""
    if not bars:
        return bars
    lo, hi = bars[0].index, bars[-1].index
    have = {b.index: b for b in bars}
    out = []
    for i in range(lo, hi + 1):
        b = have.get(i)
        if b is None:
            b = QBar(index=i, subdivision=4, grid_name="16th")
            b.start_time = float(grid.to_time(grid.bar_zero_beat + i * bpb))
            b.end_time = float(grid.to_time(grid.bar_zero_beat + (i + 1) * bpb))
        out.append(b)
    return out


def _detect_swing(hits, pos) -> tuple[bool, float]:
    """Off-beat 8ths sitting near 2/3 of the beat mean the feel is swung."""
    frac = pos - np.floor(pos)
    cym = np.array([h.inst in ("hihat", "ride", "openhh") for h in hits])
    off = (frac > 0.35) & (frac < 0.85) & cym
    if off.sum() < 8:
        return False, 0.5
    med = float(np.median(frac[off]))
    # 0.5 = straight, 0.667 = full triplet swing
    if med > 0.58:
        return True, med
    return False, med
