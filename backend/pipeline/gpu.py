"""One place to decide where the neural stages run.

A 3 GB GTX 1060 runs Demucs ~10x faster than 10 CPU threads, but only one
model fits at a time: every GPU inference takes LOCK, and the other worker
keeps going on CPU-side stages meanwhile. DRUMS_DEVICE=cpu forces CPU."""
import os
import threading

LOCK = threading.Lock()


def device() -> str:
    forced = os.environ.get("DRUMS_DEVICE")
    if forced:
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
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
