"""Plain in-memory counters (F7). No Prometheus — the plan says not to let
tooling eat the time; a JSON endpoint is enough to prove the point."""
import time
from collections import deque
from threading import Lock


class Metrics:
    def __init__(self, max_samples: int = 2000):
        self._lock = Lock()
        self._latencies_ms: deque[float] = deque(maxlen=max_samples)
        self.total_inspections = 0
        self.total_rejections = 0
        self.last_success_ts: float | None = None
        self.rollback_events: list[dict] = []

    def record(self, latency_ms: float, rejected: bool) -> None:
        with self._lock:
            self._latencies_ms.append(latency_ms)
            self.total_inspections += 1
            if rejected:
                self.total_rejections += 1
            self.last_success_ts = time.time()

    def record_rollback(self, from_version: str, to_version: str, reason: str) -> None:
        with self._lock:
            self.rollback_events.append({
                "ts": time.time(),
                "from_version": from_version,
                "to_version": to_version,
                "reason": reason,
            })

    def percentiles(self) -> dict:
        with self._lock:
            return self._percentiles_locked()

    def _percentiles_locked(self) -> dict:
        samples = sorted(self._latencies_ms)
        if not samples:
            return {"p50": None, "p95": None, "p99": None}
        return {
            "p50": _percentile(samples, 0.50),
            "p95": _percentile(samples, 0.95),
            "p99": _percentile(samples, 0.99),
        }

    def snapshot(self) -> dict:
        with self._lock:
            time_since_success = (
                time.time() - self.last_success_ts if self.last_success_ts else None
            )
            return {
                "total_inspections": self.total_inspections,
                "total_rejections": self.total_rejections,
                "seconds_since_last_success": time_since_success,
                "rollback_events": list(self.rollback_events),
                **self._percentiles_locked(),
            }


def _percentile(sorted_samples: list[float], pct: float) -> float:
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    idx = min(int(pct * len(sorted_samples)), len(sorted_samples) - 1)
    return sorted_samples[idx]


metrics = Metrics()
