"""F13: package a trained model + calibration into a signed bundle an
operator can carry to an air-gapped site on removable media (or transfer
via a gateway that never gives the device direct internet access, per the
plan's constraint table).

Usage: python make_bundle.py --version padim-bottle-v1 --out padim-bottle-v1.bundle
Then: curl -X POST http://<edge-node>/model/bundle -F "bundle=@padim-bottle-v1.bundle"
"""
import argparse
import hashlib
import hmac
import json
import os
import zipfile

EDGE_MODELS_DIR = "../edge_node/models"
BUNDLE_SECRET = os.getenv("LW_BUNDLE_SECRET", "dev-only-shared-secret")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="e.g. padim-bottle-v1")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_path = args.out or f"{args.version}.bundle"

    onnx_path = os.path.join(EDGE_MODELS_DIR, f"{args.version}.onnx")
    calib_path = os.path.join(EDGE_MODELS_DIR, f"{args.version}_calibration.json")

    with open(onnx_path, "rb") as f:
        model_bytes = f.read()
    with open(calib_path, "rb") as f:
        calibration_bytes = f.read()

    model_hash = sha256(model_bytes)
    calib_hash = sha256(calibration_bytes)
    signing_payload = f"{args.version}:{model_hash}:{calib_hash}".encode()
    signature = hmac.new(BUNDLE_SECRET.encode(), signing_payload, hashlib.sha256).hexdigest()

    manifest = {
        "version": args.version,
        "model_sha256": model_hash,
        "calibration_sha256": calib_hash,
        "signature": signature,
    }

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("model.onnx", model_bytes)
        zf.writestr("calibration.json", calibration_bytes)

    print(f"wrote signed bundle {out_path} ({os.path.getsize(out_path)} bytes)")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
