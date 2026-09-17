"""One-off: download only the MVTec AD 'bottle' category from the Voxel51
HF mirror, using per-file URLs instead of the 5.27GB combined archive.
SSL verification disabled because this sandbox's clock is set to 2026,
making some intermediate CA certs appear expired for otherwise-valid HTTPS
endpoints -- not appropriate for anything credentialed, fine for a public
research dataset with a known checksum-free but well-known source.
"""
import argparse
import json
import os
import random
import urllib.request
import ssl

SAMPLES_URL = "https://huggingface.co/datasets/Voxel51/mvtec-ad/resolve/main/samples.json"
RAW_BASE = "https://huggingface.co/datasets/Voxel51/mvtec-ad/resolve/main/"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return resp.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--train-subsample", type=int, default=80)
    args = parser.parse_args()
    out_dir = f"datasets/mvtec_{args.category}"

    print(f"fetching samples.json for category={args.category} ...")
    samples = json.loads(fetch(SAMPLES_URL))["samples"]
    matched = [s for s in samples if s["category"]["label"] == args.category]

    train_good = [s for s in matched if s["split"] == "train" and s["defect"]["label"] == "good"]
    test_good = [s for s in matched if s["split"] == "test" and s["defect"]["label"] == "good"]
    test_defect = [s for s in matched if s["split"] == "test" and s["defect"]["label"] != "good"]

    random.Random(42).shuffle(train_good)
    train_good = train_good[:args.train_subsample]

    groups = {
        "train/good": train_good,
        "test/good": test_good,
        "test/defect": test_defect,
    }

    for subdir, items in groups.items():
        out = os.path.join(out_dir, subdir)
        os.makedirs(out, exist_ok=True)
        for i, s in enumerate(items):
            url = RAW_BASE + s["filepath"]
            defect_label = s["defect"]["label"]
            fname = f"{i:03d}_{defect_label}.png"
            dest = os.path.join(out, fname)
            if os.path.exists(dest):
                continue
            data = fetch(url)
            with open(dest, "wb") as f:
                f.write(data)
        print(f"{subdir}: {len(items)} images")

    print("done")


if __name__ == "__main__":
    main()
