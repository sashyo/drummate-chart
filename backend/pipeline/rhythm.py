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
