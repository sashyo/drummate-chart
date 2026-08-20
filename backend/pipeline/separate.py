"""Stage 2 - isolate the drum kit from the full mix.

Uses Demucs (htdemucs) when available, which gives a genuinely clean drum
stem and makes everything downstream far more accurate. Falls back to a
cheap harmonic/percussive split so the app still works without torch.

Besides the mono signal used for analysis this also renders two files you
can actually listen to:
  drums.mp3     - the isolated kit, to hear what was transcribed
  backing.mp3   - everything *except* the kit, to play along to
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 44100


@dataclass
class Stems:
    mono: np.ndarray                    # drum stem, mono, 44.1 kHz - for analysis
    method: str
    files: dict = field(default_factory=dict)   # name -> Path


def _cache_key(wav: Path, model: str) -> str:
    st = wav.stat()
    return hashlib.sha1(f"{wav}|{st.st_size}|{int(st.st_mtime)}|{model}".encode()).hexdigest()[:16]


def demucs_available() -> bool:
    try:
        import demucs.apply  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _write_mp3(stereo: np.ndarray, path: Path, sr: int = SR, bitrate: str = "192k") -> None:
    """stereo: (2, n) float32 in roughly [-1, 1]."""
    import soundfile as sf
    peak = float(np.abs(stereo).max())
    if peak > 1.0:
        stereo = stereo / peak
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), stereo.T, sr, subtype="PCM_16")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp),
         "-codec:a", "libmp3lame", "-b:a", bitrate, str(path)],
        check=True, capture_output=True)
    tmp.unlink(missing_ok=True)


def drum_stem(wav: Path, out_dir: Path, cache_dir: Path, progress=None,
              model: str = "htdemucs", render_audio: bool = True) -> Stems:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / f"drums_{_cache_key(wav, model)}.npy"
    drums_mp3 = out_dir / "drums.mp3"
    backing_mp3 = out_dir / "backing.mp3"

    cached_stereo = cache_dir / f"stereo_{_cache_key(wav, model)}.npz"
    if npy.exists() and (not render_audio or (drums_mp3.exists() and backing_mp3.exists())):
        if progress:
            progress(0.45, "Drum stem (cached)")
        files = {}
        if drums_mp3.exists():
            files["drums"] = drums_mp3
        if backing_mp3.exists():
            files["backing"] = backing_mp3
        return Stems(mono=np.load(npy), method=model, files=files)

    if demucs_available():
        drums_st, backing_st = _demucs(wav, progress, model)
        method = model
    else:
        drums_st, backing_st = _hpss_fallback(wav, progress)
        method = "hpss-fallback"

    mono = drums_st.mean(axis=0).astype(np.float32)
    np.save(npy, mono)

    files = {}
    if render_audio:
        if progress:
            progress(0.46, "Rendering isolated drum track")
        _write_mp3(drums_st, drums_mp3)
        files["drums"] = drums_mp3
        if backing_st is not None:
            _write_mp3(backing_st, backing_mp3)
            files["backing"] = backing_mp3
    if cached_stereo.exists():
        cached_stereo.unlink(missing_ok=True)

    return Stems(mono=mono, method=method, files=files)


def _demucs(wav: Path, progress, model: str):
    """Return (drums_stereo, everything_else_stereo) as (2, n) float arrays."""
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if progress:
        progress(0.15, "Loading separation model")
    net = get_model(model)
    net.eval()

    # fetch.py already hands us 44.1 kHz stereo PCM, so soundfile is enough
    # and we avoid pulling in torchaudio just to read a wav.
    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    data = data.T                                   # (channels, samples)
    if sr != net.samplerate:
        import librosa
        data = np.stack([
            librosa.resample(c, orig_sr=sr, target_sr=net.samplerate) for c in data
        ])
    if data.shape[0] == 1:
        data = np.repeat(data, 2, axis=0)
    audio = torch.from_numpy(np.ascontiguousarray(data[: net.audio_channels]))

    ref = audio.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    audio_n = (audio - mean) / std

    chunks = max(1, int(np.ceil(audio.shape[-1] / net.samplerate / 30.0)))
    state = {"n": 0}

    class _Cb:
        def __call__(self, d):
            if d.get("state") == "end":
                state["n"] += 1
                if progress:
                    progress(0.15 + 0.28 * min(1.0, state["n"] / (chunks * 2)),
                             "Separating drums from the mix")

    # the fine-tuned bag is the quality path - give it the better overlap too
    overlap = 0.25 if model.endswith("_ft") else 0.15
    with torch.no_grad():
        sources = apply_model(net, audio_n[None], device="cpu", split=True,
                              overlap=overlap, progress=False, callback=_Cb())[0]
    sources = sources * std + mean

    di = net.sources.index("drums")
    drums = sources[di].cpu().numpy()
    backing = (sources.sum(dim=0) - sources[di]).cpu().numpy()

    if net.samplerate != SR:
        import librosa
        drums = np.stack([librosa.resample(c, orig_sr=net.samplerate, target_sr=SR) for c in drums])
        backing = np.stack([librosa.resample(c, orig_sr=net.samplerate, target_sr=SR) for c in backing])
    return drums.astype(np.float32), backing.astype(np.float32)


def _hpss_fallback(wav: Path, progress):
    import librosa
    if progress:
        progress(0.20, "Separating percussion (fallback)")
    y, _ = librosa.load(str(wav), sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    perc, harm = [], []
    for ch in y:
        h, p = librosa.effects.hpss(ch, margin=(1.0, 3.0))
        perc.append(p)
        harm.append(h)
    if progress:
        progress(0.43, "Percussion isolated")
    return np.stack(perc).astype(np.float32), np.stack(harm).astype(np.float32)
