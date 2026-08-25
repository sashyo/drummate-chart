"""Files a transcription creates from the user's audio, so the server can
delete them the moment the job is done. Per worker thread."""
import threading
from pathlib import Path

_local = threading.local()


def note(path) -> None:
    lst = getattr(_local, "files", None)
    if lst is None:
        lst = _local.files = []
    lst.append(Path(path))


def consume() -> list:
    lst = getattr(_local, "files", None) or []
    _local.files = []
    return lst
