"""Stage 3 - tempo, beat grid, meter and downbeats.

Everything downstream works in *beat space* rather than seconds, so a song
that drifts in tempo still lands on a sane grid. `BeatGrid.to_beats` maps
seconds -> fractional beats by interpolating the tracked beat times.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 44100
HOP = 256


@dataclass
class BeatGrid:
    tempo: float
    beat_times: np.ndarray     # seconds, one per beat
    beats_per_bar: int
    downbeat_index: int        # index into beat_times of the first downbeat
    tempo_curve: np.ndarray    # instantaneous bpm per beat

    def to_beats(self, t):
        """Seconds -> fractional beat number (0 = first tracked beat)."""
        t = np.asarray(t, dtype=float)
        bt = self.beat_times
        if len(bt) < 2:
            return t * self.tempo / 60.0
        idx = np.arange(len(bt), dtype=float)
        # Linear interpolation inside, constant-tempo extrapolation outside.
        first_ibi = bt[1] - bt[0]
        last_ibi = bt[-1] - bt[-2]
        out = np.interp(t, bt, idx)
        lo = t < bt[0]
        hi = t > bt[-1]
        if np.any(lo):
            out = np.where(lo, (t - bt[0]) / max(first_ibi, 1e-6), out)
        if np.any(hi):
            out = np.where(hi, (len(bt) - 1) + (t - bt[-1]) / max(last_ibi, 1e-6), out)
        return out

    def to_time(self, b):
        b = np.asarray(b, dtype=float)
        bt = self.beat_times
        if len(bt) < 2:
            return b * 60.0 / self.tempo
        idx = np.arange(len(bt), dtype=float)
        first_ibi = bt[1] - bt[0]
        last_ibi = bt[-1] - bt[-2]
        out = np.interp(b, idx, bt)
        lo = b < 0
        hi = b > len(bt) - 1
        out = np.where(lo, bt[0] + b * first_ibi, out)
        out = np.where(hi, bt[-1] + (b - (len(bt) - 1)) * last_ibi, out)
        return out

    @property
    def bar_zero_beat(self) -> float:
        return float(self.downbeat_index)


def analyse(y: np.ndarray, mix: np.ndarray | None = None, progress=None,
            beats_per_bar: int = 4, fixed_tempo: float | None = None,
            lock_grid: bool = False) -> BeatGrid:
    import librosa

    if progress:
        progress(0.48, "Tracking tempo and beats")

    # Beat tracking is more stable on the full mix (bass + comping give it
    # harmonic anchors); fall back to the drum stem when no mix is supplied.
    track_on = mix if mix is not None and len(mix) == len(y) else y
    onset_env = librosa.onset.onset_strength(y=track_on, sr=SR, hop_length=HOP, aggregate=np.median)

    if fixed_tempo:
        tempo = float(fixed_tempo)
        _, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=SR, hop_length=HOP, bpm=tempo, trim=False)
    else:
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=SR, hop_length=HOP, trim=False)
        tempo = float(np.atleast_1d(tempo)[0])
        tempo = _fix_octave(tempo)
        _, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=SR, hop_length=HOP, bpm=tempo, trim=False)

    beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)

    if lock_grid or fixed_tempo:
        # Clock-time grid: one constant BPM for the whole song, as played to a
        # click. The tracked beats only pin down the tempo and the phase; the
        # grid itself is rigid, so quantise variance comes purely from the
        # player, not from beat-tracker wobble.
        bpm = float(fixed_tempo or tempo)
        beat_times = _locked_grid(onset_env, bpm, len(y) / SR)
        if progress:
            progress(0.55, f"Locked grid at {bpm:.1f} BPM")
    if len(beat_times) < 4:  # pathological input - synthesise a grid
        dur = len(y) / SR
        step = 60.0 / (tempo or 120.0)
        beat_times = np.arange(0.0, max(dur, step * 4), step)

    ibi = np.diff(beat_times)
    tempo_curve = np.concatenate([[60.0 / ibi[0]], 60.0 / np.maximum(ibi, 1e-6)])

    downbeat = _find_downbeat(y, beat_times, beats_per_bar)

    if progress:
        progress(0.55, f"Tempo {tempo:.1f} BPM")

    return BeatGrid(
        tempo=float(np.median(tempo_curve)),
        beat_times=beat_times,
        beats_per_bar=beats_per_bar,
        downbeat_index=downbeat,
        tempo_curve=tempo_curve,
    )


def _locked_grid(onset_env: np.ndarray, bpm: float, duration: float) -> np.ndarray:
    """Constant-tempo beat grid; the phase is fitted to the onsets.

    Scores a cycle of candidate offsets by summing onset strength at each
    grid line and keeps the best - i.e. the grid alignment real hits agree
    with most.
    """
    import librosa
    period = 60.0 / bpm
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=SR, hop_length=HOP)
    best_phase, best = 0.0, -1.0
    for phase in np.linspace(0, period, 48, endpoint=False):
        grid = np.arange(phase, duration, period)
        idx = np.searchsorted(times, grid)
        idx = idx[idx < len(onset_env)]
        score = float(onset_env[idx].sum())
        if score > best:
            best, best_phase = score, phase
    return np.arange(best_phase, max(duration, best_phase + 4 * period), period)


def _align_score(times: np.ndarray, period: float, phase: float, sigma: float = 0.09) -> float:
    frac = ((times - phase) / period) % 1.0
    d = np.minimum(frac, 1.0 - frac)
    return float(np.exp(-(d / sigma) ** 2).sum())


def _best_octave(hits, period: float, bpb: int) -> float:
    """Choose among T/2, T, 2T by how musical the resulting bars are.

    At the true tempo snares concentrate on two beat classes (the backbeat)
    while every beat class carries hits; at half tempo snares smear across
    all four classes and most kicks sit 'off the beat'; at double tempo
    half the beat classes are empty. Autocorrelation alone cannot tell
    these apart - a drum & bass track at 172 reads as 86 with a clean grid.
    """
    kick = np.array([h.time for h in hits if h.inst == "kick"])
    snare = np.array([h.time for h in hits if h.inst == "snare"])
    anchors = np.concatenate([kick, snare])
    if len(anchors) < 16 or bpb != 4:
        return period

    def judge(T0):
        # refine the candidate's period first: a 0.3% error drifts two
        # beats across a song and unfairly smears the finer tempo
        best_ph, best_sc, T = 0.0, -1.0, T0
        for TT in np.linspace(T0 * 0.98, T0 * 1.02, 21):
            for ph in np.linspace(0, TT, 48, endpoint=False):
                sc = _align_score(anchors, TT, ph)
                if sc > best_sc:
                    best_sc, best_ph, T = sc, ph, TT
        pos = (anchors - best_ph) / T
        on = np.abs(pos - np.round(pos)) < 0.15
        on_grid = float(on.mean())
        cls = (np.round(pos[on]).astype(int)) % bpb
        is_snare = np.concatenate([np.zeros(len(kick), bool), np.ones(len(snare), bool)])[on]
        counts = np.bincount(cls, minlength=bpb) / max(1, on.sum())
        populated = int(np.sum(counts >= 0.08))
        if is_snare.sum() >= 4:
            sc_counts = np.sort(np.bincount(cls[is_snare], minlength=bpb))[::-1]
            top2 = float(sc_counts[:2].sum() / max(1, is_snare.sum()))
            # a backbeat is TWO comparable classes; at double tempo every
            # snare collapses into one class (top1 ~ 1, top2 ~ 0)
            balance = float(min(sc_counts[0], sc_counts[1]) / max(1, sc_counts[0]))
        else:
            top2, balance = 0.5, 0.5
        # top2 only counts once it clearly exceeds the 'snare on every beat'
        # spread (0.5); balance is the backbeat's defining property; on-grid
        # is down-weighted because finer grids always score higher on it
        backbeat = 2.0 * max(0.0, top2 - 0.5) / 0.5
        return ((2.0 if populated >= 4 else 1.0 if populated == 3 else 0.0)
                + backbeat + 2.0 * balance + 0.5 * on_grid)

    cands = [T for T in (period / 2, period, period * 2) if 60 / 200 <= T <= 60 / 60]
    if not cands:
        return period
    scored = [(judge(T), T) for T in cands]
    return max(scored, key=lambda x: x[0])[1]


def grid_from_drums(hits, duration: float, beats_per_bar: int,
                    tempo_hint: float | None = None, fixed_tempo: float | None = None,
                    progress=None) -> BeatGrid | None:
    """A beat grid derived from the drums, not from the mix.

    Generic beat trackers lock onto guitar chugs and off-beat hats and drift
    between the true tempo and 4:3 / 2:1 errors; every chart built on such a
    grid is rhythmically scrambled even when each hit is right. Kick and
    snare onsets are a far better witness: their autocorrelation gives the
    period, a joint period/phase search aligns the grid, and a windowed
    phase track follows slow drift.
    """
    anchors = np.array(sorted(h.time for h in hits if h.inst in ("kick", "snare")))
    if len(anchors) < 16:
        return None

    # 1. period from the onset-train autocorrelation (50-200 BPM)
    res = 0.005
    train = np.zeros(int(duration / res) + 2)
    train[np.clip((anchors / res).astype(int), 0, len(train) - 1)] = 1.0
    ac = np.correlate(train, train, "full")[len(train) - 1:]
    lags = np.arange(len(ac)) * res
    m = (lags >= 60 / 200) & (lags <= 60 / 50)
    if fixed_tempo:
        period = 60.0 / fixed_tempo
    else:
        cand_lags = lags[m]
        cand = ac[m] * np.exp(-((np.log(cand_lags) - np.log(60 / 110)) ** 2) / (2 * 0.5 ** 2))
        period = float(cand_lags[int(np.argmax(cand))])
        # octave sanity against the tracker: same tempo family, drums win
        while period < 60 / 200:
            period *= 2
        while period > 60 / 60:
            period /= 2

    # 1b. octave: pick the tempo whose bars look like bars
    if not fixed_tempo:
        period = _best_octave(hits, period, beats_per_bar)

    # 2. joint period/phase refinement
    best = (-1.0, period, 0.0)
    span = 0.0 if fixed_tempo else 0.035
    for T in np.linspace(period * (1 - span), period * (1 + span), 1 if fixed_tempo else 29):
        for ph in np.linspace(0, T, 64, endpoint=False):
            sc = _align_score(anchors, T, ph)
            if sc > best[0]:
                best = (sc, T, ph)
    _, T, ph = best

    # 3. slow drift: local phase offset per window, unwrapped and interpolated.
    # Windows are a fixed 12 s (not N beats) so a fast tempo doesn't get
    # short, jumpy windows; each step is capped at 0.12 beat so the phase
    # can follow a drummer but can never creep onto the off-beat. The
    # drifted grid is only kept when it is DECISIVELY better than the
    # constant one - click-steady tracks get a rigid grid.
    n_beats = int(np.ceil(duration / T)) + 2
    k = np.arange(n_beats)
    base = ph + k * T
    win = 12.0
    centers = np.arange(win / 2, duration, win / 2)
    offs, ts = [], []
    prev = 0.0
    for c in centers:
        sel = anchors[(anchors >= c - win / 2) & (anchors < c + win / 2)]
        if len(sel) < 8:
            continue
        frac = ((sel - ph) / T) % 1.0
        cands = np.linspace(-0.3, 0.3, 61)
        scores = [np.exp(-((np.minimum((frac - d) % 1.0, 1 - (frac - d) % 1.0)) / 0.09) ** 2).sum()
                  for d in cands]
        d = float(cands[int(np.argmax(scores))])
        d = prev + float(np.clip(d - prev, -0.12, 0.12))
        offs.append(d); ts.append(c); prev = d
    beat_times = base
    if ts:
        delta = np.interp(base, ts, offs)
        drifted = base + delta * T
        # score both grids on the anchors; keep drift only if it clearly wins
        def grid_score(bt):
            idx = np.interp(anchors, bt, np.arange(len(bt)))
            dd = np.abs(idx - np.round(idx))
            return float(np.exp(-(dd / 0.09) ** 2).sum())
        if grid_score(drifted) > 1.06 * grid_score(base):
            beat_times = drifted
    beat_times = beat_times[beat_times >= 0]
    if beat_times[0] > T:
        beat_times = np.concatenate([np.arange(beat_times[0] - T, -1e-9, -T)[::-1], beat_times])

    ibi = np.diff(beat_times)
    tempo_curve = np.concatenate([[60.0 / T], 60.0 / np.maximum(ibi, 1e-6)])
    grid = BeatGrid(tempo=60.0 / T, beat_times=beat_times, beats_per_bar=beats_per_bar,
                    downbeat_index=0, tempo_curve=tempo_curve)
    if progress:
        progress(0.80, f"Grid from the drums: {60 / T:.1f} BPM")
    return grid


def refine_with_hits(grid: BeatGrid, hits, progress=None) -> BeatGrid:
    """Align the beat grid to the drums, then pick the downbeat musically.

    Beat tracking on a full mix often locks onto guitar chugs or off-beat
    hats, leaving every correct hit written a fraction of a beat off - a
    chart that is 'right' and unplayable. Kick and snare land on quarter
    notes far more than off them in nearly all popular music, so: slide
    the grid (fraction of a beat) to maximise on-beat kick+snare mass, then
    choose the bar phase where snares sit on 2 & 4 and kicks on 1 & 3.
    """
    anchors = np.array([h.time for h in hits if h.inst in ("kick", "snare")])
    if len(anchors) < 8 or len(grid.beat_times) < 4:
        return grid

    pos = np.asarray(grid.to_beats(anchors), dtype=float)
    frac = pos % 1.0
    best_shift, best = 0.0, -1.0
    for shift in np.linspace(0, 1, 96, endpoint=False):
        d = np.abs(((frac - shift + 0.5) % 1.0) - 0.5)
        score = float(np.exp(-(d / 0.09) ** 2).sum())
        if score > best:
            best, best_shift = score, shift
    # a shift near 1.0 is a shift near 0 the other way
    if best_shift > 0.5:
        best_shift -= 1.0

    idx = np.arange(len(grid.beat_times), dtype=float) + best_shift
    new_times = np.asarray(grid.to_time(idx), dtype=float)
    if best_shift < 0:
        new_times = new_times[new_times > 0.0]
    grid = BeatGrid(tempo=grid.tempo, beat_times=new_times,
                    beats_per_bar=grid.beats_per_bar, downbeat_index=0,
                    tempo_curve=np.resize(grid.tempo_curve, len(new_times)))

    # downbeat: backbeat prior (4/4 and 2/4-style meters), else energy-based
    bpb = grid.beats_per_bar
    pos = np.asarray(grid.to_beats(anchors), dtype=float)
    on = np.abs(pos - np.round(pos)) < 0.15
    beat_idx = np.round(pos[on]).astype(int)
    insts = np.array([h.inst for h in hits if h.inst in ("kick", "snare")])[on]
    if bpb in (4, 2) and len(beat_idx) >= 8:
        best_p, best_s = 0, -1e9
        for p in range(bpb):
            rel = (beat_idx - p) % bpb
            if bpb == 4:
                sc = (np.sum((insts == "snare") & np.isin(rel, [1, 3]))
                      + 0.6 * np.sum((insts == "kick") & np.isin(rel, [0, 2]))
                      - 0.8 * np.sum((insts == "snare") & (rel == 0)))
            else:
                sc = np.sum((insts == "snare") & (rel == 1)) + 0.6 * np.sum((insts == "kick") & (rel == 0))
            if sc > best_s:
                best_s, best_p = sc, p
        grid.downbeat_index = int(best_p)
    if progress:
        progress(0.80, f"Grid aligned to the drums (shift {best_shift:+.2f} beat)")
    return grid


def _fix_octave(tempo: float) -> float:
    """Nudge obvious half/double-time errors into a drummer-friendly range."""
    t = tempo
    while t < 65:
        t *= 2
    while t > 190:
        t /= 2
    return t


def _find_downbeat(y: np.ndarray, beat_times: np.ndarray, bpb: int) -> int:
    """Pick the beat phase whose bass-drum / crash weight is heaviest."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    low = S[(freqs >= 30) & (freqs <= 130)].sum(axis=0)
    high = S[freqs >= 6000].sum(axis=0)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=SR, hop_length=HOP)

    def weight(env, t, win=0.06):
        i0 = np.searchsorted(times, t - 0.01)
        i1 = np.searchsorted(times, t + win)
        return float(env[i0:i1].max()) if i1 > i0 else 0.0

    n = len(beat_times)
    if n < bpb * 2:
        return 0
    lows = np.array([weight(low, t) for t in beat_times])
    highs = np.array([weight(high, t) for t in beat_times])
    lows /= (lows.max() or 1.0)
    highs /= (highs.max() or 1.0)
    score = lows + 0.45 * highs

    best_phase, best = 0, -1.0
    for p in range(bpb):
        sel = score[p::bpb]
        if len(sel) == 0:
            continue
        s = float(sel.mean())
        if s > best:
            best, best_phase = s, p
    return best_phase
