"""Runtime configuration for the edge node. Threshold and a few knobs are
mutable at runtime (F3); everything else is set at process start via env vars."""
import os
from datetime import time
from threading import Lock


class Config:
    def __init__(self):
        self._lock = Lock()
        self.threshold: float = float(os.getenv("LW_THRESHOLD", "0.5"))
        # F2: "inline" runs inference on the event loop (deliberately wrong,
        # for measuring the collapse); "worker" runs it in a process pool.
        self.inference_mode: str = os.getenv("LW_INFERENCE_MODE", "worker")
        self.worker_count: int = int(os.getenv("LW_WORKER_COUNT", "2"))
        self.model_dir: str = os.getenv("LW_MODEL_DIR", "./models")
        self.fleet_url: str = os.getenv("LW_FLEET_URL", "")
        self.device_id: str = os.getenv("LW_DEVICE_ID", "site-1")
        self.port: int = int(os.getenv("LW_PORT", "8000"))
        # what this node tells the fleet to reach it at -- "localhost" only
        # works when everything runs as local processes; in Docker each
        # container needs its own service/hostname here instead.
        self.advertise_host: str = os.getenv("LW_ADVERTISE_HOST", "localhost")
        # F13: shared-secret HMAC key for verifying local signed model
        # bundles. A real deployment would use asymmetric signing + a
        # distributed public key, not a shared secret -- see bundle.py.
        self.bundle_secret: str = os.getenv("LW_BUNDLE_SECRET", "dev-only-shared-secret")
        # F9: maintenance window, local time, HH:MM-HH:MM. Empty = always open.
        self.maintenance_window: str = os.getenv("LW_MAINTENANCE_WINDOW", "")
        # F8: rollback ceiling on holdout false-positive rate.
        self.rollback_fp_ceiling: float = float(os.getenv("LW_ROLLBACK_FP_CEILING", "0.05"))
        self.holdout_dir: str = os.getenv("LW_HOLDOUT_DIR", "../data/holdout")

    def get_threshold(self) -> float:
        with self._lock:
            return self.threshold

    def set_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        with self._lock:
            self.threshold = value

    def in_maintenance_window(self, now: time | None = None) -> bool:
        if not self.maintenance_window:
            return True
        now = now or _local_now()
        start_s, end_s = self.maintenance_window.split("-")
        start = _parse_hhmm(start_s)
        end = _parse_hhmm(end_s)
        if start <= end:
            return start <= now <= end
        return now >= start or now <= end  # window wraps midnight


def _parse_hhmm(s: str) -> time:
    h, m = s.strip().split(":")
    return time(int(h), int(m))


def _local_now() -> time:
    from datetime import datetime
    return datetime.now().time()


config = Config()
