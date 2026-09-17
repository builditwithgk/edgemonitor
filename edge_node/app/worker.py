"""F2: keep CPU-bound inference off the async event loop.

LW_INFERENCE_MODE=inline runs score_image() directly inside the async
handler — deliberately wrong, kept as a mode so its throughput collapse
under concurrent load can be measured and written to FINDINGS.md.

LW_INFERENCE_MODE=worker (default) runs it in a ProcessPoolExecutor via
loop.run_in_executor, so the event loop stays free to accept new requests
while inference is running.
"""
import asyncio
from concurrent.futures import ProcessPoolExecutor

from .config import config
from .inference import score_image

_executor: ProcessPoolExecutor | None = None


def start_pool() -> None:
    global _executor
    if config.inference_mode == "worker" and _executor is None:
        _executor = ProcessPoolExecutor(max_workers=config.worker_count)


def stop_pool() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


async def run_inference(image_bytes: bytes, model_version: str) -> tuple[float, float]:
    if config.inference_mode == "inline":
        # Blocks the event loop for the full inference duration.
        return score_image(image_bytes, model_version)

    loop = asyncio.get_running_loop()
    if _executor is None:
        start_pool()
    return await loop.run_in_executor(_executor, score_image, image_bytes, model_version)
