"""Train PaDiM on an MVTec AD category (good samples only, matching the real
constraint that defect data doesn't exist), export to ONNX, and -- for the
default 'bottle' category only -- refresh the edge node's holdout set.

Same clock/cert caveat as download_bottle.py: this sandbox's system clock is
set to 2026, which makes some otherwise-valid intermediate CA certs appear
expired. The one network call this script makes -- fetching the pretrained
ResNet18 backbone weights -- needs verification relaxed for the same reason.
Nothing credentialed touches this path; it's a public, well-known model
checkpoint. If your system clock is correct, delete the `ssl._create_default_
https_context` line below and this will "just work" with normal verification.

Usage: python train_padim.py [--category bottle|screw|...] [--version-name padim-bottle-v1]
"""
import argparse
import json
import shutil
import ssl
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context  # see docstring

import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim

EDGE_MODELS_DIR = Path("../edge_node/models")
HOLDOUT_DIR = Path("../data/holdout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--version-name", default=None,
                         help="output filename stem, default padim-<category>-v1")
    args = parser.parse_args()
    version_name = args.version_name or f"padim-{args.category}-v1"
    dataset_root = Path(f"datasets/mvtec_{args.category}")

    datamodule = Folder(
        name=args.category,
        root=dataset_root,
        normal_dir="train/good",
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        train_batch_size=1,   # plan §3a: batch size 1, memory-constrained CPU
        eval_batch_size=1,
        num_workers=0,        # default of 8 spawns 8 full processes on Windows -- OOM on 8GB RAM
    )

    model = Padim(backbone="resnet18", layers=["layer1", "layer2"])
    engine = Engine(max_epochs=1, accelerator="cpu")

    print(f"fitting PaDiM on '{args.category}' (forward passes only, no gradient training)...")
    engine.fit(model=model, datamodule=datamodule)

    print("running test predictions for score separation check...")
    results = engine.predict(model=model, datamodule=datamodule)

    good_scores, defect_scores = [], []
    for batch in results:
        scores = batch.pred_score.flatten().tolist()
        labels = batch.gt_label.flatten().tolist()  # 0 = good, 1 = defect
        for s, l in zip(scores, labels):
            (defect_scores if l else good_scores).append(float(s))

    if good_scores:
        print(f"good test scores:   n={len(good_scores)}  "
              f"min={min(good_scores):.4f} max={max(good_scores):.4f} "
              f"mean={sum(good_scores)/len(good_scores):.4f}")
    if defect_scores:
        print(f"defect test scores: n={len(defect_scores)}  "
              f"min={min(defect_scores):.4f} max={max(defect_scores):.4f} "
              f"mean={sum(defect_scores)/len(defect_scores):.4f}")

    # anomalib's Engine.export() bakes the post-processor's min-max
    # normalization into the ONNX graph, but as of anomalib 2.6 that graph
    # does not reproduce the live model's calibration (verified: identical
    # preprocessed input gives PyTorch raw score ~46 but the exported graph
    # saturates to 1.0). Sidestepping it: export the raw model only, do the
    # min-max normalization ourselves in edge_node/app/inference.py using
    # calibration stats saved alongside the model.
    class ScoreOnly(torch.nn.Module):
        def __init__(self, padim_model):
            super().__init__()
            self.padim_model = padim_model

        def forward(self, x):
            return self.padim_model(x).pred_score

    EDGE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = EDGE_MODELS_DIR / f"{version_name}.onnx"
    dummy_input = torch.zeros(1, 3, 256, 256)
    torch.onnx.export(
        ScoreOnly(model.model).eval(),
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["pred_score"],
        dynamic_axes={"input": {0: "batch_size"}, "pred_score": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,  # dynamo=True's verbose logging prints a unicode checkmark
    )                   # that crashes on Windows' cp1252 console encoding
    print(f"exported raw-score ONNX model to {onnx_path}")

    calibration = {
        "image_min": model.post_processor.image_min.item(),
        "image_max": model.post_processor.image_max.item(),
    }
    calib_path = EDGE_MODELS_DIR / f"{version_name}_calibration.json"
    calib_path.write_text(json.dumps(calibration, indent=2))
    print(f"saved calibration {calibration} to {calib_path}")

    if args.category == "bottle":
        # Refresh the edge node's holdout set with real known-good bottle
        # images (the ones F7/F8 evaluate the *active* model against).
        if HOLDOUT_DIR.exists():
            for f in HOLDOUT_DIR.glob("*"):
                f.unlink()
        HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
        holdout_source = dataset_root / "test" / "good"
        for f in holdout_source.glob("*.png"):
            shutil.copy(f, HOLDOUT_DIR / f.name)
        print(f"copied {len(list(HOLDOUT_DIR.glob('*')))} real holdout images to {HOLDOUT_DIR}")


if __name__ == "__main__":
    main()
