"""F10/F13: push a heartbeat to the fleet service on an interval. If the
fleet is unreachable (air-gap simulation), this must fail silently and the
node must keep inspecting normally — heartbeat health is best-effort, never
a dependency of the inspection path."""
import asyncio

import httpx

from .config import config
from .holdout import evaluate_false_positive_rate
from .metrics import metrics
from .model_store import model_store

_task: asyncio.Task | None = None


async def _loop(interval_s: float) -> None:
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            await _send_once(client)
            await asyncio.sleep(interval_s)


async def _send_once(client: httpx.AsyncClient) -> None:
    if not config.fleet_url:
        return
    try:
        fp_rate = evaluate_false_positive_rate(model_store.active.version)
        snap = metrics.snapshot()
        await client.post(
            f"{config.fleet_url}/devices/heartbeat",
            json={
                "device_id": config.device_id,
                "url": f"http://{config.advertise_host}:{config.port}",
                "active_version": model_store.active.version,
                "previous_version": model_store.previous.version if model_store.previous else None,
                "holdout_fp_rate": fp_rate,
                "p50_ms": snap.get("p50"),
                "total_inspections": snap.get("total_inspections", 0),
            },
        )
    except httpx.HTTPError:
        pass  # air-gapped or fleet down — inspection continues regardless


def start(interval_s: float = 10.0) -> None:
    global _task
    if config.fleet_url and _task is None:
        _task = asyncio.create_task(_loop(interval_s))


def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
