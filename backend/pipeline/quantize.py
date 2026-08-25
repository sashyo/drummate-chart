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
    beats: int = 4

    @property
    def ticks(self) -> int:
        return PPQ * self.beats


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
    n_est = int(np.ceil(max(rel.max(), 0) / bpb)) + 2
    bounds = grid.bar_bounds(n_est) - grid.bar_zero_beat    # bar starts, beats since bar 0
    bar_len = np.diff(bounds)
    bar_idx = np.searchsorted(bounds, rel, side="right") - 1
    neg = bar_idx < 0                            # before the first downbeat
    bar_idx = np.where(neg, np.floor(rel / bpb).astype(int), bar_idx)
    bar_beats = np.where(neg, bpb, bar_len[np.clip(bar_idx, 0, len(bar_len) - 1)]).astype(int)
    bar_off = np.where(neg, bar_idx * bpb, bounds[np.clip(bar_idx, 0, len(bounds) - 1)])
    pos = rel - bar_off                         # 0..beats within the bar

    def bar_span(i: int) -> tuple[float, float]:
        if i < 0:
            return grid.bar_zero_beat + i * bpb, grid.bar_zero_beat + (i + 1) * bpb
        j = min(i, len(bounds) - 2)
        b0 = bounds[j] + (i - j) * bpb
        return grid.bar_zero_beat + b0, grid.bar_zero_beat + b0 + (bar_len[j] if i == j else bpb)

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

        # De-lag: a laid-back or pushed bar sits at a roughly constant offset
        # from the grid; without removing it, relaxed 8ths match the triplet
        # grid better than the straight one. The offset is SEARCHED (a naive
        # median over a dense 16th bar is bimodal and lands mid-grid) and only
        # applied when the shifted bar really is coarse-grid-consistent.
        p2 = p - _bar_lag(p)

        prev_n = bars[int(b) - 1].subdivision if int(b) - 1 in bars else None
        best_n = _choose_subdiv(p2, allowed, prev_n)
        bar = QBar(index=int(b), subdivision=best_n, grid_name=GRIDS[best_n][0],
                   beats=int(bar_beats[sel[0]]))

        step = 1.0 / best_n
        for k, i in enumerate(sel):
            snapped_beat = float(np.round(p2[k] / step) * step)
            tick = int(round(snapped_beat * PPQ))
            tick = max(0, min(bar.ticks, tick))
            h = hits[i]
            bar.notes.append(QNote(
                tick=tick, inst=h.inst, velocity=float(h.velocity),
                ghost=h.ghost, accent=h.accent, time=float(h.time)))
        bars[int(b)] = bar

    # A note snapped to the very end of a bar belongs to the next downbeat.
    for b in sorted(bars):
        carry = [n for n in bars[b].notes if n.tick >= bars[b].ticks]
        if carry:
            bars[b].notes = [n for n in bars[b].notes if n.tick < bars[b].ticks]
            nb0, nb1 = bar_span(b + 1)
            nxt = bars.setdefault(b + 1, QBar(index=b + 1, subdivision=bars[b].subdivision,
                                              grid_name=bars[b].grid_name, beats=int(round(nb1 - nb0))))
            for n in carry:
                n.tick = 0
                nxt.notes.append(n)

    ordered = [bars[b] for b in sorted(bars)]
    for bar in ordered:
        bar.notes = _merge(bar.notes)
        bar.notes.sort(key=lambda n: (n.tick, n.inst))
        b0, b1 = bar_span(bar.index)
        bar.start_time = float(grid.to_time(b0))
        bar.end_time = float(grid.to_time(b1))
        bar.beats = int(round(b1 - b0))

    _cleanup(ordered, bpb)
    ordered = _fill_empty_bars(ordered, grid, bpb, bar_span)
    ordered = _trim_edges(ordered)

    return QScore(bars=ordered, ppq=PPQ, beats_per_bar=bpb, ticks_per_bar=ticks_per_bar,
                  swing=swing, swing_ratio=ratio, tempo=grid.tempo)


def _bar_lag(p: np.ndarray, limit: float = 0.2) -> float:
    if len(p) < 3:
        return 0.0
    cost0 = float(np.median(_dist(p, 0.5)))
    best_off, best_cost = 0.0, cost0
    for off in np.linspace(-limit, limit, 41):
        cost = float(np.median(_dist(p - off, 0.5)))
        if cost < best_cost - 1e-9:
            best_cost, best_off = cost, off
    # Apply only when shifting is DECISIVELY better than staying put and the
    # result is genuinely 8th-grid-consistent - zero-mean jitter can always
    # shave a little off the median, and chasing that shaved every bar onto
    # noise-fit offsets. 16th and triplet bars never pass at any offset.
    if best_cost < 0.05 and best_cost < cost0 - 0.04 and abs(best_off) >= 0.06:
        return best_off
    return 0.0


def _dist(p: np.ndarray, step: float) -> np.ndarray:
    return np.abs(((p + step / 2) % step) - step / 2)


def _choose_subdiv(p: np.ndarray, allowed: list[int],
                   prev_n: int | None = None) -> int:
    """Pick the coarsest grid the evidence justifies.

    Aggregate snap error flips jittery 8ths onto phantom 16th (or triplet)
    grids, because extra slots absorb noise. Instead each finer grid must be
    DEMANDED by hits that are close to one of its exclusive slots and far
    from every coarser slot. A single dead-centre hit (a real pickup) is
    enough; borderline hits need two.
    """
    if len(p) == 0:
        return 4 if 4 in allowed else allowed[0]
    d8, d16, d32 = _dist(p, .5), _dist(p, .25), _dist(p, .125)
    d3, d6 = _dist(p, 1 / 3), _dist(p, 1 / 6)

    def votes(near, far_pairs, tight, strong_tight):
        far = np.ones(len(p), dtype=bool)
        for d, m in far_pairs:
            far &= d > m
        weak = int(np.sum((near < tight) & far))
        strong = int(np.sum((near < strong_tight) & far
                            & (far_pairs[0][0] > far_pairs[0][1] + 0.04)))
        return weak, strong

    def demanded(w, st, n):
        if st >= 1:
            return True
        # Anti-flicker hysteresis: switching to a finer grid than the
        # previous bar used takes three borderline hits, not two - a real
        # 8th->16th transition brings six or more.
        need = 3 if (prev_n is not None and n > prev_n) else 2
        return w >= need

    c32 = votes(d32, [(d16, 0.09)], 0.05, 0.035)
    c6 = votes(d6, [(d16, 0.09), (d3, 0.06)], 0.05, 0.035)
    c16 = votes(d16, [(d8, 0.16)], 0.08, 0.055)
    c3 = votes(d3, [(d8, 0.15), (d16, 0.08)], 0.07, 0.05)

    if 8 in allowed and demanded(*c32, 8):
        return 8
    if 6 in allowed and demanded(*c6, 6) and c6[0] > c16[0]:
        return 6
    if 3 in allowed and demanded(*c3, 3) and c3[0] > c16[0]:
        return 3
    if 4 in allowed and demanded(*c16, 4):
        return 4
    return 2 if 2 in allowed else allowed[0]


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


def _cleanup(bars: list[QBar], bpb_default: int) -> None:
    """Remove detection jitter so the same groove spells the same way.

    One repair, deliberately conservative: when the cymbal line fills >=80%%
    of a bar's grid, a lone one-slot hole inside the run is a detection miss -
    a drummer does not skip one 16th mid-run. Genuinely sparse patterns
    (offbeat hats fill 50%%) never reach the threshold and are left alone.
    (An earlier "echo removal" rule was measured to eat real alternating-
    velocity 16ths and was dropped; same-tick doubles are merged upstream.)
    """
    # A kick at the velocity floor (below its own local 10th percentile) that
    # sits exactly under a snare stroke is snare bleeding into the kick stem,
    # not a foot: nobody notates a kick ghost. Only when the song's kicks are
    # otherwise strong - a genuinely soft kick line keeps its notes.
    kv = [x.velocity for b in bars for x in b.notes if x.inst == "kick"]
    if kv and float(np.median(kv)) >= 0.3:
        kslots = [{x.tick for x in b.notes if x.inst == "kick"} for b in bars]
        # Is a floor kick under the snare bleed, or the groove? Bleed shows
        # on a minority of snares (Every Breath You Take: 12%); a kick that
        # doubles the backbeat is on most of them (Another One Bites the
        # Dust: 65%, played at floor velocity because the snare masks it).
        n_sn = sum(1 for b in bars for x in b.notes if x.inst == "snare")
        n_co = sum(1 for b in bars for x in b.notes if x.inst == "kick"
                   and any(y.inst == "snare" and y.tick == x.tick for y in b.notes))
        doubling = n_sn and n_co / n_sn >= 0.5
        # a bar position the foot never plays with any weight, anywhere in
        # the song, is bleed however regularly it shows (Highway to Hell's
        # '&' of 2: 36 bars at velocity 0.09 against real kicks at 0.9)
        by_slot: dict[int, list[float]] = {}
        for b in bars:
            for x in b.notes:
                if x.inst == "kick":
                    by_slot.setdefault(x.tick, []).append(x.velocity)
        weak_slot = {t for t, v in by_slot.items() if len(v) >= 4 and float(np.median(v)) <= 0.12}
        for bi, bar in enumerate(bars):
            snare_ticks = {x.tick for x in bar.notes if x.inst == "snare"}
            near = set()
            if bi:
                near |= kslots[bi - 1]
            if bi + 1 < len(bars):
                near |= kslots[bi + 1]
            # a floor kick under a snare is bleed; a floor kick that repeats
            # nowhere nearby is a stray (a feathered kick pattern repeats)
            bar.notes = [x for x in bar.notes if not (
                x.inst == "kick" and x.velocity <= 0.12
                and ((x.tick in snare_ticks and not doubling) or x.tick not in near
                     or (x.tick in weak_slot and not (doubling and x.tick in snare_ticks))))]
            # ...and when the song doubles its backbeat, a snare on a beat
            # with no kick under it is the detector losing the kick in the
            # snare, not the drummer resting the foot: complete the pattern
            if doubling:
                have = {x.tick for x in bar.notes if x.inst == "kick"}
                kvel = float(np.median([x.velocity for b in bars for x in b.notes
                                        if x.inst == "kick" and any(
                                            y.inst == "snare" and y.tick == x.tick for y in b.notes)] or [0.3]))
                for sn in [x for x in bar.notes if x.inst == "snare" and x.tick % PPQ == 0
                           and x.tick not in have and not x.ghost]:
                    bar.notes.append(QNote(tick=sn.tick, inst="kick", velocity=kvel, time=sn.time))
                bar.notes.sort(key=lambda x: (x.tick, x.inst))

    # A floor-velocity snare (below its own local 10th percentile) is either
    # a ghost note or bleed. Ghost grooves REPEAT - the same slot carries a
    # snare in the bar before or after - while bleed lands anywhere. Keep
    # the repeating ones, drop the strays (published charts write neither
    # Billie Jean's nor Every Breath's 0.05-velocity blips).
    sv = [x.velocity for b in bars for x in b.notes if x.inst == "snare"]
    if sv and float(np.median(sv)) >= 0.3:
        slots = [{x.tick for x in b.notes if x.inst == "snare"} for b in bars]
        for bi, bar in enumerate(bars):
            near = set()
            if bi:
                near |= slots[bi - 1]
            if bi + 1 < len(bars):
                near |= slots[bi + 1]
            kick_beats = {x.tick for x in bar.notes if x.inst == "kick" and x.tick % PPQ == 0}
            # ...and a floor snare exactly on a kicked quarter-note beat is
            # bleed however often it repeats: ghosts live between the beats
            bar.notes = [x for x in bar.notes if not (
                x.inst == "snare" and x.velocity <= 0.06
                and (x.tick not in near or x.tick in kick_beats))]

    for bi, bar in enumerate(bars):
        bpb = bar.beats
        n = bar.subdivision
        slot = PPQ // n
        cym = [x for x in bar.notes if x.inst in CYMS]
        if not cym:
            continue
        span = bar.end_time - bar.start_time
        vel = float(np.median([x.velocity for x in cym]))
        prev_cym = [x.tick for x in bars[bi - 1].notes if x.inst in CYMS] if bi else []
        next_cym = [x.tick for x in bars[bi + 1].notes if x.inst in CYMS] if bi + 1 < len(bars) else []
        # Judge the run at the hats' OWN pulse. An 8th-note hat line fills
        # only half of a 16th grid, so a 16th-only test never repaired the
        # most common groove in rock.
        for pulse in sorted({PPQ // 2, slot}, reverse=True):
            nslots = (PPQ * bpb) // pulse
            on = [x for x in cym if x.tick % pulse == 0]
            have = {x.tick // pulse for x in on}
            if nslots and len(have) / nslots >= 0.75:
                # A run does not stop at the bar line: the neighbour on the
                # far side of it is the next bar's downbeat / the previous
                # bar's last stroke. The last up-beat of a bar is the softest
                # stroke in many grooves and was the hole this never closed.
                last = PPQ * bpb - pulse
                ext = set(have)
                if next_cym and 0 in next_cym:
                    ext.add(nslots)
                if prev_cym and last in prev_cym:
                    ext.add(-1)
                for sl in range(0, nslots):
                    if sl in have or (sl - 1) not in ext or (sl + 1) not in ext:
                        continue
                    bar.notes.append(QNote(
                        tick=sl * pulse, inst="hihat", velocity=vel,
                        time=bar.start_time + span * sl / nslots))
                # a run at the 8th pulse: stray, weaker hats on odd 16ths are
                # doubles or bleed, not playing
                # (a bar of true 16th hats has only half its cymbals on the
                # 8th pulse and is left alone; 8 hats + 3 strays is 73%)
                if pulse == PPQ // 2 and len(on) >= 0.6 * len(cym):
                    bar.notes = [x for x in bar.notes if not (
                        x.inst in CYMS and x.tick % pulse != 0 and x.velocity < 0.6 * vel)]
                break
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


def _fill_empty_bars(bars: list[QBar], grid, bpb: int, bar_span=None) -> list[QBar]:
    """Insert genuinely empty bars so the chart keeps its bar numbering."""
    if not bars:
        return bars
    lo, hi = bars[0].index, bars[-1].index
    have = {b.index: b for b in bars}
    out = []
    for i in range(lo, hi + 1):
        b = have.get(i)
        if b is None:
            if bar_span is not None:
                b0, b1 = bar_span(i)
            else:
                b0, b1 = grid.bar_zero_beat + i * bpb, grid.bar_zero_beat + (i + 1) * bpb
            b = QBar(index=i, subdivision=4, grid_name="16th", beats=int(round(b1 - b0)))
            b.start_time = float(grid.to_time(b0))
            b.end_time = float(grid.to_time(b1))
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
