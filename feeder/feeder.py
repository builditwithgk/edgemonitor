"""F4: the line feeder. Pushes images at a configured rate to an edge node's
/inspect endpoint, optionally injecting a defect proportion, and reports
observed throughput plus p50/p95/p99 latency.

Stage 1-3 note: the stub inference function scores purely off image bytes,
so "defect rate" here just picks which of two image pools to draw from —
swap in real MVTec good/defect folders once the dataset is in place.
"""
import argparse
import asyncio
import glob
import os
import random
import time

import httpx


def _synthetic_image(seed: int, defective: bool) -> bytes:
    """Deterministic filler bytes for early testing before the MVTec dataset
    is wired in. Defective images get a marker byte so scores differ."""
    rng = random.Random(seed)
    body = bytes(rng.getrandbits(8) for _ in range(256))
    marker = b"\xff" if defective else b"\x00"
    return marker + body


def _load_pool(image_dir: str | None, defective: bool, count: int) -> list[bytes]:
    if image_dir:
        subdir = os.path.join(image_dir, "defect" if defective else "good")
        files = sorted(glob.glob(os.path.join(subdir, "*")))
        if files:
            images = []
            for f in files:
                with open(f, "rb") as fh:
                    images.append(fh.read())
            return images
    return [_synthetic_image(i, defective) for i in range(count)]


async def _send_one(client: httpx.AsyncClient, url: str, image_bytes: bytes) -> tuple[float, bool]:
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/inspect",
            files={"image": ("part.bin", image_bytes, "application/octet-stream")},
        )
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, True
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, False


async def run_feeder(url: str, rate_per_min: float, duration_s: int, defect_rate: float,
                      image_dir: str | None, concurrency: int) -> dict:
    good_pool = _load_pool(image_dir, defective=False, count=100)
    defect_pool = _load_pool(image_dir, defective=True, count=20)

    interval_s = 60.0 / rate_per_min
    latencies: list[float] = []
    failures = 0
    total = 0

    sem = asyncio.Semaphore(concurrency)
    results_lock = asyncio.Lock()

    async def worker(client: httpx.AsyncClient, image_bytes: bytes):
        nonlocal failures, total
        async with sem:
            elapsed_ms, ok = await _send_one(client, url, image_bytes)
        async with results_lock:
            latencies.append(elapsed_ms)
            total += 1
            if not ok:
                failures += 1

    start_time = time.perf_counter()
    tasks = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.perf_counter() - start_time < duration_s:
            pool = defect_pool if random.random() < defect_rate else good_pool
            image_bytes = random.choice(pool)
            tasks.append(asyncio.create_task(worker(client, image_bytes)))
            await asyncio.sleep(interval_s)
        await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - start_time

    latencies.sort()
    return {
        "requests": total,
        "failures": failures,
        "wall_seconds": wall_s,
        "observed_throughput_per_min": (total / wall_s) * 60 if wall_s else 0,
        "p50_ms": _pct(latencies, 0.50),
        "p95_ms": _pct(latencies, 0.95),
        "p99_ms": _pct(latencies, 0.99),
    }


def _pct(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(int(p * len(sorted_vals)), len(sorted_vals) - 1)
    return round(sorted_vals[idx], 2)


def main():
    parser = argparse.ArgumentParser(description="Edgemonitor feeder")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rate", type=float, default=600, help="images per minute")
    parser.add_argument("--duration", type=int, default=60, help="seconds")
    parser.add_argument("--defect-rate", type=float, default=0.01)
    parser.add_argument("--image-dir", default=None, help="folder with good/ and defect/ subfolders")
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    result = asyncio.run(run_feeder(
        url=args.url,
        rate_per_min=args.rate,
        duration_s=args.duration,
        defect_rate=args.defect_rate,
        image_dir=args.image_dir,
        concurrency=args.concurrency,
    ))
    print(result)


if __name__ == "__main__":
    main()
