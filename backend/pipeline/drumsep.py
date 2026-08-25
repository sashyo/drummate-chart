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


def _content_key(wav: Path) -> str:
    """Hash the audio CONTENT - a path/mtime key misses the cache on every
    re-run of the same song through a temp file."""
    import soundfile as sf
    try:
        y, _ = sf.read(str(wav), dtype="int16", always_2d=True)
        return hashlib.sha1(y[:: max(1, len(y) // 200000)].tobytes()
                            + str(len(y)).encode()).hexdigest()[:16]
    except Exception:
        st = wav.stat()
        return hashlib.sha1(f"{wav}|{st.st_size}".encode()).hexdigest()[:16]


def separate(wav: Path, cache_dir: Path, progress=None) -> dict[str, np.ndarray]:
    """Return {stem: mono float32 @44.1k} for the drum audio in `wav`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _content_key(wav) + "-" + hashlib.sha1(MODEL.encode()).hexdigest()[:6]
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
        # The split blocks for roughly 1x realtime on CPU. Project progress
        # from elapsed time so the bar keeps moving and shows a time estimate.
        import threading, time, soundfile as sf
        try:
            info = sf.info(str(wav))
            from . import gpu
            # ~1x realtime on 10 CPU threads; a few times faster on a GPU
            expect = max(20.0, info.duration * (0.3 if gpu.device() == "cuda" else 1.15))
        except Exception:
            expect = 240.0
        stop = threading.Event()

        def ticker():
            t0 = time.time()
            while not stop.wait(2.0):
                el = time.time() - t0
                frac = min(0.97, el / expect)
                left = int(expect - el)
                if progress:
                    if left > 3:
                        progress(0.50 + 0.10 * frac,
                                 f"Splitting the kit: kick / snare / toms / hats / cymbals \u2014 about {left // 60}:{left % 60:02d} left")
                    else:
                        progress(0.50 + 0.10 * frac,
                                 f"Splitting the kit \u2014 taking longer than estimated (server is busy), still working \u2014 {int(el) // 60}:{int(el) % 60:02d} elapsed")
        if progress:
            progress(0.50, "Splitting the kit: kick / snare / toms / hats / cymbals")
        th = threading.Thread(target=ticker, daemon=True); th.start()
        try:
            from . import gpu
            if gpu.device() == "cuda":
                try:
                    with gpu.Turn(progress, "Splitting the kit"):
                        outs = sep.separate(str(wav))
                except Exception as exc:  # noqa: BLE001 - 3 GB card: retry with short segments
                    if not gpu.is_oom(exc):
                        raise
                    import torch
                    torch.cuda.empty_cache()
                    print(f"drumsep: GPU out of memory ({exc}); retrying with segment 128")
                    sep = Separator(
                        log_level=40, output_dir=str(tmp), output_format="WAV",
                        model_file_dir=os.path.expanduser("~/.cache/drumsep"),
                        mdxc_params={"segment_size": 128, "override_model_segment_size": True,
                                     "batch_size": 1, "overlap": 2, "pitch_shift": 0},
                    )
                    sep.load_model(model_filename=MODEL)
                    with gpu.Turn(progress, "Splitting the kit"):
                        outs = sep.separate(str(wav))
            else:
                outs = sep.separate(str(wav))
        finally:
            stop.set(); th.join(timeout=3)
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
