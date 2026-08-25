"""One place to decide where the neural stages run.

A 3 GB GTX 1060 runs Demucs ~10x faster than 10 CPU threads, but only one
model fits at a time: every GPU inference takes LOCK, and the other worker
keeps going on CPU-side stages meanwhile. DRUMS_DEVICE=cpu forces CPU."""
import os
import threading

LOCK = threading.Lock()

# First come, first served. threading.Lock hands the GPU to whichever waiter
# wakes first, so a job could be leapfrogged by every newer job; with two
# workers that showed as one chart sitting on "Loading separation model"
# for ten minutes. Tickets are served in order, and the waiter is told.
_turn = threading.Condition()
_next_ticket = 0
_now_serving = 0


class Turn:
    def __init__(self, progress=None, stage: str = ""):
        self.progress, self.stage = progress, stage

    def __enter__(self):
        global _next_ticket
        with _turn:
            me = _next_ticket; _next_ticket += 1
            waited = False
            while me != _now_serving:
                if not waited and self.progress:
                    self.progress(None, f"{self.stage} \u2014 waiting for the GPU (another chart is using it)")
                waited = True
                _turn.wait(timeout=2.0)
        return self

    def __exit__(self, *exc):
        global _now_serving
        with _turn:
            _now_serving += 1
            _turn.notify_all()
        return False


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
