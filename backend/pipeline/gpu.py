"""One place to decide where the neural stages run.

A 3 GB GTX 1060 runs Demucs ~10x faster than 10 CPU threads, but only one
model fits at a time: every GPU inference takes LOCK, and the other worker
keeps going on CPU-side stages meanwhile. DRUMS_DEVICE=cpu forces CPU."""
import os
import threading

LOCK = threading.Lock()


_tuned = False


def device() -> str:
    global _tuned
    forced = os.environ.get("DRUMS_DEVICE")
    try:
        import torch
        if forced:
            dev = forced
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda" and not _tuned:
            # cuDNN's workspace alone failed to initialise on a 3 GB card
            # shared with the desktop (CUDNN_STATUS_NOT_INITIALIZED); plain
            # convolutions are a little slower and need a fraction of the
            # memory. Measured: Demucs 21 s and DrumSep 21 s per 45 s of
            # audio on a GTX 1060 3 GB, against 94 s and 489 s on 10 CPU
            # threads under load.
            torch.backends.cudnn.enabled = vram_gb() >= 6.0
            _tuned = True
        return dev
    except Exception:  # noqa: BLE001
        return "cpu"


def vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
