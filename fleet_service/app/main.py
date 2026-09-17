from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from . import db
from .orchestrator import run_staged_rollout


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Edgemonitor Fleet Service", lifespan=lifespan)


# --------------------------------------------------------------------- F10 --

class Heartbeat(BaseModel):
    device_id: str
    url: str
    active_version: str
    previous_version: str | None = None
    holdout_fp_rate: float | None = None
    p50_ms: float | None = None
    total_inspections: int = 0


@app.post("/devices/heartbeat")
async def heartbeat(hb: Heartbeat):
    db.upsert_device(
        device_id=hb.device_id, url=hb.url, active_version=hb.active_version,
        previous_version=hb.previous_version, holdout_fp_rate=hb.holdout_fp_rate,
        p50_ms=hb.p50_ms, total_inspections=hb.total_inspections,
    )
    return {"received": True}


@app.get("/devices")
async def devices():
    return db.list_devices()


@app.get("/devices/{device_id}")
async def device(device_id: str):
    d = db.get_device(device_id)
    if d is None:
        raise HTTPException(status_code=404, detail="unknown device")
    return d


# --------------------------------------------------------------------- F12 --

@app.get("/drift")
async def drift():
    """Sites whose holdout FP rate has moved materially from their own
    baseline (captured at first heartbeat). See db.upsert_device."""
    return [d for d in db.list_devices() if d["drifting"]]


# --------------------------------------------------------------------- F11 --

class RolloutRequest(BaseModel):
    version: str
    stages: list[list[str]]  # e.g. [["site-1"], ["site-2", "site-3"]]


@app.post("/rollout")
async def start_rollout(req: RolloutRequest, background_tasks: BackgroundTasks):
    rollout_id = db.create_rollout(req.version, f"queued, {len(req.stages)} stage(s)")
    background_tasks.add_task(run_staged_rollout, req.version, req.stages, rollout_id)
    return {"rollout_id": rollout_id, "status": "in_progress"}


@app.get("/rollout/{rollout_id}")
async def rollout_status(rollout_id: int):
    r = db.get_rollout(rollout_id)
    if r is None:
        raise HTTPException(status_code=404, detail="unknown rollout")
    return r
