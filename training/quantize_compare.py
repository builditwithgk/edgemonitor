"""F15: quantize a trained PaDiM ONNX model to INT8 (dynamic quantization --
no calibration dataset needed, weights only) and compare FP32 vs INT8 on
this machine: latency (p50/p95 over repeated real inference) and accuracy
(score separation + false-positive rate on the real holdout set).

Usage: python quantize_compare.py --version padim-bottle-v1
"""
import argparse
import glob
import json
import os
import statistics
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image

EDGE_MODELS_DIR = "../edge_node/models"
HOLDOUT_DIR = "../data/holdout"
DEFECT_DIR = "datasets/mvtec_bottle/test/defect"
IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - MEAN) / STD
    return arr[np.newaxis, ...]


def bench(session: ort.InferenceSession, input_name: str, tensors: list[np.ndarray],
          repeats: int) -> dict:
    latencies = []
    scores = []
    for _ in range(repeats):
        for t in tensors:
            start = time.perf_counter()
            out = session.run(None, {input_name: t})
            latencies.append((time.perf_counter() - start) * 1000)
            scores.append(float(out[0].reshape(-1)[0]))
    latencies.sort()
    n = len(latencies)
    return {
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)] if n > 1 else latencies[0],
        "raw_scores": scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="padim-bottle-v1")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    fp32_path = os.path.join(EDGE_MODELS_DIR, f"{args.version}.onnx")
    int8_path = os.path.join(EDGE_MODELS_DIR, f"{args.version}-int8.onnx")
    calib_path = os.path.join(EDGE_MODELS_DIR, f"{args.version}_calibration.json")
    with open(calib_path) as f:
        calibration = json.load(f)

    print(f"quantizing {fp32_path} -> {int8_path} (dynamic, weights only)...")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    fp32_size = os.path.getsize(fp32_path)
    int8_size = os.path.getsize(int8_path)
    print(f"size: FP32 {fp32_size/1e6:.1f}MB -> INT8 {int8_size/1e6:.1f}MB "
          f"({(1 - int8_size/fp32_size)*100:.1f}% smaller)")

    good_paths = sorted(glob.glob(os.path.join(HOLDOUT_DIR, "*.png")))
    defect_paths = sorted(glob.glob(os.path.join(DEFECT_DIR, "*.png")))
    good_tensors = [preprocess(p) for p in good_paths]
    defect_tensors = [preprocess(p) for p in defect_paths]

    def normalize(raw_scores):
        span = calibration["image_max"] - calibration["image_min"]
        return [min(max((s - calibration["image_min"]) / span, 0.0), 1.0) for s in raw_scores]

    results = {}
    for label, path in [("FP32", fp32_path), ("INT8", int8_path)]:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        good_bench = bench(sess, input_name, good_tensors, args.repeats)
        defect_bench = bench(sess, input_name, defect_tensors, args.repeats)

        good_scores = normalize(good_bench["raw_scores"])
        defect_scores = normalize(defect_bench["raw_scores"])
        all_latencies = sorted(
            [l for l in [good_bench["p50_ms"], defect_bench["p50_ms"]]]
        )

        results[label] = {
            "size_mb": round(os.path.getsize(path) / 1e6, 1),
            "p50_ms": round(statistics.median([good_bench["p50_ms"], defect_bench["p50_ms"]]), 2),
            "good_score_range": (round(min(good_scores), 4), round(max(good_scores), 4)),
            "defect_score_range": (round(min(defect_scores), 4), round(max(defect_scores), 4)),
            "good_score_mean": round(statistics.mean(good_scores), 4),
            "defect_score_mean": round(statistics.mean(defect_scores), 4),
        }
        print(f"{label}: {results[label]}")

    print("\n--- summary ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
