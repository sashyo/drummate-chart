"""Stage 2b - split the drum stem into per-drum stems (neural).

MDX23C DrumSep (aufr33 / jarredou) separates a drum stem into kick, snare,
toms, hi-hat, ride and crash. With each drum on its own track, detection
becomes onset picking per stem instead of spectral guesswork - the single
biggest accuracy lever in the pipeline. Costs ~1x realtime on CPU.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
STEMS = ("kick", "snare", "toms", "hh", "ride", "crash")
SR = 44100


def available() -> bool:
    try:
        import audio_separator  # noqa: F401
        return True
    except Exception:
        return False


def _shim_librosa():
    """audio-separator still calls librosa.get_duration(filename=...)."""
    import librosa
    if getattr(librosa.get_duration, "_shimmed", False):
        return
    orig = librosa.get_duration

    def get_duration(*a, filename=None, **k):
        if filename is not None:
            k["path"] = filename
        return orig(*a, **k)
    get_duration._shimmed = True
    librosa.get_duration = get_duration


def separate(wav: Path, cache_dir: Path, progress=None) -> dict[str, np.ndarray]:
    """Return {stem: mono float32 @44.1k} for the drum audio in `wav`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    st = wav.stat()
    key = hashlib.sha1(f"{wav}|{st.st_size}|{int(st.st_mtime)}|{MODEL}".encode()).hexdigest()[:16]
    cached = cache_dir / f"drumsep_{key}.npz"
    if cached.exists():
        z = np.load(cached)
        return {k: z[k] for k in STEMS if k in z}

    _shim_librosa()
    import librosa
    from audio_separator.separator import Separator

    if progress:
        progress(0.47, "Loading kit-splitting model")
    tmp = Path(tempfile.mkdtemp(prefix="drumsep_"))
    try:
        sep = Separator(
            log_level=40, output_dir=str(tmp), output_format="WAV",
            model_file_dir=os.path.expanduser("~/.cache/drumsep"),
            mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                         "batch_size": 1, "overlap": 2, "pitch_shift": 0},
        )
        sep.load_model(model_filename=MODEL)
        if progress:
            progress(0.50, "Splitting the kit: kick / snare / toms / hats / cymbals")
        outs = sep.separate(str(wav))
        stems: dict[str, np.ndarray] = {}
        for o in outs:
            name = o.split("_(")[-1].split(")")[0]
            if name in STEMS:
                y, _ = librosa.load(str(tmp / o), sr=SR, mono=True)
                stems[name] = y.astype(np.float32)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    np.savez(cached, **stems)
    return stems
