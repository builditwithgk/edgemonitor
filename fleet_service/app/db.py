"""SQLite store for the fleet. Kept boring on purpose (§5) — the interest is
in rollout/drift logic, not the persistence layer."""
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = "fleet.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    active_version TEXT,
    previous_version TEXT,
    last_heartbeat_ts REAL,
    holdout_fp_rate REAL,
    baseline_fp_rate REAL,
    p50_ms REAL,
    total_inspections INTEGER,
    drifting INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rollouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    status TEXT NOT NULL,
    created_ts REAL,
    updated_ts REAL,
    detail TEXT
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def upsert_device(device_id: str, url: str, active_version: str, previous_version: str | None,
                   holdout_fp_rate: float | None, p50_ms: float | None,
                   total_inspections: int) -> None:
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT baseline_fp_rate FROM devices WHERE device_id = ?", (device_id,)
        ).fetchone()
        baseline = existing["baseline_fp_rate"] if existing else holdout_fp_rate
        if baseline is None:
            baseline = holdout_fp_rate

        drifting = 0
        if baseline is not None and holdout_fp_rate is not None:
            drifting = int(abs(holdout_fp_rate - baseline) > 0.15)

        conn.execute(
            """INSERT INTO devices
                 (device_id, url, active_version, previous_version, last_heartbeat_ts,
                  holdout_fp_rate, baseline_fp_rate, p50_ms, total_inspections, drifting)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(device_id) DO UPDATE SET
                 url=excluded.url, active_version=excluded.active_version,
                 previous_version=excluded.previous_version,
                 last_heartbeat_ts=excluded.last_heartbeat_ts,
                 holdout_fp_rate=excluded.holdout_fp_rate,
                 p50_ms=excluded.p50_ms, total_inspections=excluded.total_inspections,
                 drifting=excluded.drifting""",
            (device_id, url, active_version, previous_version, time.time(),
             holdout_fp_rate, baseline, p50_ms, total_inspections, drifting),
        )


def list_devices() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM devices").fetchall()
        return [dict(r) for r in rows]


def get_device(device_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,)).fetchone()
        return dict(row) if row else None


def create_rollout(version: str, detail: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO rollouts (version, status, created_ts, updated_ts, detail) "
            "VALUES (?, 'in_progress', ?, ?, ?)",
            (version, time.time(), time.time(), detail),
        )
        return cur.lastrowid


def update_rollout(rollout_id: int, status: str, detail: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE rollouts SET status = ?, updated_ts = ?, detail = ? WHERE id = ?",
            (status, time.time(), detail, rollout_id),
        )


def get_rollout(rollout_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM rollouts WHERE id = ?", (rollout_id,)).fetchone()
        return dict(row) if row else None
