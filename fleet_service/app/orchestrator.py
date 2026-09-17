"""F11: staged rollout. Promotion between stages is gated on the target
device(s) reporting healthy after activation — not a timer. Any rollback
(reported by the edge node itself, per F8) halts the rest of the rollout.
"""
import httpx

from . import db


async def promote_to_devices(client: httpx.AsyncClient, version: str,
                              device_ids: list[str]) -> list[dict]:
    """Stage + activate `version` on each device in this cohort. Returns one
    result dict per device with whatever the edge node reported."""
    results = []
    for device_id in device_ids:
        device = db.get_device(device_id)
        if device is None:
            results.append({"device_id": device_id, "ok": False, "reason": "unknown device"})
            continue
        try:
            await client.post(f"{device['url']}/model", json={"version": version})
            resp = await client.post(f"{device['url']}/model/activate")
            resp.raise_for_status()
            body = resp.json()
            healthy = not body.get("rolled_back", False)
            results.append({"device_id": device_id, "ok": healthy, **body})
        except httpx.HTTPError as e:
            results.append({"device_id": device_id, "ok": False, "reason": str(e)})
    return results


async def run_staged_rollout(version: str, stages: list[list[str]], rollout_id: int) -> None:
    """Runs stage by stage. A stage only proceeds to the next if every device
    in it activated healthily. On any failure, the rollout halts immediately
    and later stages are never attempted."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        for i, cohort in enumerate(stages):
            db.update_rollout(rollout_id, "in_progress", f"stage {i + 1}/{len(stages)}: {cohort}")
            results = await promote_to_devices(client, version, cohort)
            if not all(r.get("ok") for r in results):
                db.update_rollout(
                    rollout_id, "halted",
                    f"stage {i + 1} failed: {results}",
                )
                return
    db.update_rollout(rollout_id, "complete", f"promoted to all {sum(len(s) for s in stages)} devices")
