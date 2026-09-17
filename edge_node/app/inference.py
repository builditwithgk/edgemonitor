"""Real inference: PaDiM (trained on MVTec AD categories, good samples only)
exported to ONNX, run via ONNX Runtime. Replaces the stage 1-3 stub -- same
call signature, so nothing upstream (API, worker pool) had to change.

Preprocessing (256x256 resize, ImageNet normalization) mirrors anomalib's
default PaDiM pre-processor exactly. The ONNX graph itself only outputs the
*raw* PaDiM score (a Mahalanobis-like distance, unbounded) -- min-max
normalization into roughly [0, 1] is done here using calibration stats
saved at training time, in `<version>_calibration.json` next to the model.
(anomalib's own Engine.export() bakes normalization into the graph too, but
as of anomalib 2.6 that baked-in version doesn't reproduce the live model's
calibration -- verified by comparing identical inputs through both paths.
Exporting the raw score and normalizing ourselves sidesteps that bug and is
arguably cleaner anyway: the decision threshold is already config, not a
constant, so this keeps normalization in the same place.)

Different model versions are different files (`<version>.onnx` +
`<version>_calibration.json` in `config.model_dir`) -- this is what lets F8's
rollback test activate a model trained on the wrong category and see it
actually behave differently, not just relabel the same weights.

Each worker process gets its own lazily-created ONNX Runtime sessions --
sessions aren't picklable, so they can't be created at import time and
shared across the ProcessPoolExecutor.
"""
import io
import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import config

IMG_SIZE = 256
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_sessions: dict[str, tuple[ort.InferenceSession, str, dict]] = {}


def _get_session(model_version: str) -> tuple[ort.InferenceSession, str, dict]:
    if model_version not in _sessions:
        onnx_path = os.path.join(config.model_dir, f"{model_version}.onnx")
        calib_path = os.path.join(config.model_dir, f"{model_version}_calibration.json")
        if not os.path.exists(onnx_path):
            # legacy fallback for the original single-model layout
            onnx_path = os.path.join(config.model_dir, "model.onnx")
            calib_path = os.path.join(config.model_dir, "calibration.json")

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        with open(calib_path) as f:
            calibration = json.load(f)
        _sessions[model_version] = (session, input_name, calibration)
    return _sessions[model_version]


def score_image(image_bytes: bytes, model_version: str) -> tuple[float, float]:
    start = time.perf_counter()
    session, input_name, calibration = _get_session(model_version)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr[np.newaxis, ...]

    outputs = session.run(None, {input_name: arr})
    raw_score = float(outputs[0].reshape(-1)[0])

    span = calibration["image_max"] - calibration["image_min"]
    normalized = (raw_score - calibration["image_min"]) / span if span else 0.0
    score = min(max(normalized, 0.0), 1.0)

    latency_ms = (time.perf_counter() - start) * 1000
    return score, latency_ms
