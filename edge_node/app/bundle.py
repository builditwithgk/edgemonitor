"""F13: apply a model update from a local signed bundle, no fleet connection
required. A bundle is a zip of {model.onnx, calibration.json, manifest.json}
where manifest.json carries the version name, sha256 of each file, and an
HMAC-SHA256 signature over those hashes.

This is deliberately not a full PKI/code-signing setup (the plan's non-goals
explicitly rule out production security hardening) -- it's a shared-secret
HMAC, enough to prove the update path (verify integrity + origin, apply
without a network round-trip to the fleet) without pretending to be a real
supply-chain security control.
"""
import hashlib
import hmac
import json
import os
import zipfile
from dataclasses import dataclass

from .config import config


class BundleError(Exception):
    pass


@dataclass
class BundleManifest:
    version: str
    model_sha256: str
    calibration_sha256: str
    signature: str

    def signing_payload(self) -> bytes:
        return f"{self.version}:{self.model_sha256}:{self.calibration_sha256}".encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_and_apply_bundle(bundle_bytes: bytes) -> str:
    """Verifies signature + content hashes, then writes the model files into
    config.model_dir under their version name. Returns the version name.
    Raises BundleError on any verification failure -- a corrupted or
    unsigned bundle must never reach the model directory."""
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(bundle_bytes)) as zf:
            manifest_raw = zf.read("manifest.json")
            model_bytes = zf.read("model.onnx")
            calibration_bytes = zf.read("calibration.json")
    except (zipfile.BadZipFile, KeyError) as e:
        raise BundleError(f"malformed bundle: {e}")

    manifest_dict = json.loads(manifest_raw)
    manifest = BundleManifest(
        version=manifest_dict["version"],
        model_sha256=manifest_dict["model_sha256"],
        calibration_sha256=manifest_dict["calibration_sha256"],
        signature=manifest_dict["signature"],
    )

    expected_sig = hmac.new(
        config.bundle_secret.encode(), manifest.signing_payload(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected_sig, manifest.signature):
        raise BundleError("signature verification failed")

    if _sha256(model_bytes) != manifest.model_sha256:
        raise BundleError("model.onnx content hash mismatch")
    if _sha256(calibration_bytes) != manifest.calibration_sha256:
        raise BundleError("calibration.json content hash mismatch")

    os.makedirs(config.model_dir, exist_ok=True)
    onnx_path = os.path.join(config.model_dir, f"{manifest.version}.onnx")
    calib_path = os.path.join(config.model_dir, f"{manifest.version}_calibration.json")
    with open(onnx_path, "wb") as f:
        f.write(model_bytes)
    with open(calib_path, "wb") as f:
        f.write(calibration_bytes)

    return manifest.version
