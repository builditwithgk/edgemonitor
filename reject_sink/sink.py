"""Stub PLC reject signal (§3a). A real PLC gets a discrete I/O pulse; here
an HTTP POST plays the same role and gets logged so the feeder/tests can
verify a reject was actually signalled."""
import time

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Edgemonitor Reject Sink")
log: list[dict] = []


class RejectSignal(BaseModel):
    device_id: str
    score: float
    model_version: str


@app.post("/reject")
async def reject(signal: RejectSignal):
    entry = {**signal.model_dump(), "ts": time.time()}
    log.append(entry)
    return {"received": True, "total_rejects": len(log)}


@app.get("/log")
async def get_log():
    return {"total_rejects": len(log), "entries": log[-100:]}
