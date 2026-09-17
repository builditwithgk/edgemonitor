"""Tracks active + previous model version (F5). Stage 1-3 only need a
version label; stage 4 will point this at real ONNX artifact paths on disk."""
from dataclasses import dataclass
from threading import Lock


@dataclass
class ModelRecord:
    version: str
    path: str | None = None  # ONNX artifact path, set from stage 4 onward


class ModelStore:
    def __init__(self, initial_version: str = "padim-bottle-v1"):
        self._lock = Lock()
        self.active = ModelRecord(version=initial_version)
        self.previous: ModelRecord | None = None
        self.staged: ModelRecord | None = None

    def stage(self, version: str, path: str | None = None) -> None:
        with self._lock:
            self.staged = ModelRecord(version=version, path=path)

    def activate_staged(self) -> ModelRecord:
        with self._lock:
            if self.staged is None:
                raise ValueError("no staged model to activate")
            self.previous = self.active
            self.active = self.staged
            self.staged = None
            return self.active

    def rollback(self) -> ModelRecord:
        with self._lock:
            if self.previous is None:
                raise ValueError("no previous model to roll back to")
            self.active, self.previous = self.previous, self.active
            return self.active

    def status(self) -> dict:
        with self._lock:
            return {
                "active_version": self.active.version,
                "previous_version": self.previous.version if self.previous else None,
                "staged_version": self.staged.version if self.staged else None,
            }


model_store = ModelStore()
