"""F14: recent rejects + operator overrides. Overrides feed the drift
signal (F12's "operator override rate" is a leading indicator of drift
that shows up before the holdout-based check catches it -- an operator
noticing the line is wrong is often faster than the next heartbeat).

Images are stored as small JPEG thumbnails, not full frames -- this is a
review queue, not an archive; keeping only the last N and downscaling
keeps memory bounded on the same constrained edge controller everything
else here runs on.
"""
import io
import itertools
import time
from collections import deque
from threading import Lock

from PIL import Image

THUMBNAIL_SIZE = (160, 160)
MAX_HISTORY = 200

_lock = Lock()
_id_counter = itertools.count(1)
_recent: deque[dict] = deque(maxlen=MAX_HISTORY)
_total_overrides = 0


def record_reject(image_bytes: bytes, score: float, model_version: str) -> None:
    thumb = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    thumb.thumbnail(THUMBNAIL_SIZE)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=70)

    with _lock:
        _recent.append({
            "id": next(_id_counter),
            "ts": time.time(),
            "score": score,
            "model_version": model_version,
            "thumbnail_jpeg": buf.getvalue(),
            "overridden": False,
        })


def list_recent() -> list[dict]:
    with _lock:
        # newest first, without the raw thumbnail bytes (fetched separately)
        return [
            {k: v for k, v in r.items() if k != "thumbnail_jpeg"}
            for r in reversed(_recent)
        ]


def get_thumbnail(review_id: int) -> bytes | None:
    with _lock:
        for r in _recent:
            if r["id"] == review_id:
                return r["thumbnail_jpeg"]
    return None


def mark_overridden(review_id: int) -> bool:
    global _total_overrides
    with _lock:
        for r in _recent:
            if r["id"] == review_id:
                if not r["overridden"]:
                    r["overridden"] = True
                    _total_overrides += 1
                return True
    return False


def override_rate() -> float | None:
    with _lock:
        if not _recent:
            return None
        return _total_overrides / len(_recent)
