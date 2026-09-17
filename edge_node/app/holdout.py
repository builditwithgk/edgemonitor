"""F7/F8: evaluate the active model against a small holdout set of known-good
images and report the false-positive rate. Used both for the /health signal
and for F8's automatic rollback decision.
"""
import glob
import os

from .config import config
from .inference import score_image


def load_holdout_bytes() -> list[bytes]:
    pattern = os.path.join(config.holdout_dir, "*")
    files = sorted(glob.glob(pattern))
    images = []
    for f in files:
        if os.path.isfile(f):
            with open(f, "rb") as fh:
                images.append(fh.read())
    return images


def evaluate_false_positive_rate(model_version: str) -> float | None:
    """All holdout images are known-good, so any score above threshold is a
    false positive. Returns None if no holdout set is present (stage 1-3)."""
    images = load_holdout_bytes()
    if not images:
        return None
    threshold = config.get_threshold()
    false_positives = 0
    for img in images:
        score, _ = score_image(img, model_version)
        if score > threshold:
            false_positives += 1
    return false_positives / len(images)
