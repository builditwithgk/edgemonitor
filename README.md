# Edgemonitor

Reference implementation of an edge visual-inspection platform for
manufacturing quality control. The detection model is a solved problem;
this project is about everything around it — safe rollout, automatic
rollback, air-gapped operation, and drift detection. See `FINDINGS.md` for
the design rationale and every number behind the claims below. This file
is just "how to run it."

Status: **The whole plan is built**, F1-F15, all four tiers, every stage
exercised with real numbers in `FINDINGS.md` — not just code that compiles.
The edge node runs a **real** PaDiM anomaly model trained on MVTec AD
'bottle' (`training/`), not a stub. Stages 6-8 (fleet, rollout, drift,
air-gap) have each been re-verified as **real Docker containers**
(`docker-compose.yml`) — three constrained (`--cpus=2 --memory=2g`)
edge-node containers plus the fleet, matching §3a's "multiple plant sites"
simulation literally. Tier 4 (F14 operator review UI, F15 FP32-vs-INT8
quantization) is done too — F15 in particular surfaced a real, useful
result: INT8 made this model both ~27x slower and broke score separation
on this hardware. See `FINDINGS.md` for the full numbers.

## Layout

```
edge_node/     FastAPI inspection API + inference worker pool (F1-F9)
fleet_service/ control plane — device registry, staged rollout, drift (F10-F12)
feeder/        line simulator + load/latency measurement (F4)
reject_sink/   stub PLC endpoint
training/      MVTec AD download + PaDiM training/ONNX export (separate venv)
data/holdout/  real known-good bottle images used for F7/F8 false-positive checks
FINDINGS.md    measured numbers, filled in as each stage is exercised
```

## Setup

Each piece is self-contained; the edge node is the one you'll run most.

```bash
cd edge_node
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Repeat for `feeder/`, `fleet_service/`, and `reject_sink/` (each has its own
`requirements.txt`).

## Run the edge node

```bash
cd edge_node
uvicorn app.main:app --port 8000
```

Try it:

```bash
curl http://localhost:8000/health
curl -X PUT http://localhost:8000/config/threshold -H "Content-Type: application/json" -d "{\"threshold\": 0.7}"
```

`/inspect` expects a multipart file upload under the field name `image`.

### Config (env vars, see `app/config.py`)

| Var | Default | Purpose |
|---|---|---|
| `LW_THRESHOLD` | 0.5 | initial decision threshold |
| `LW_INFERENCE_MODE` | worker | `inline` or `worker` — see F2 benchmark below |
| `LW_WORKER_COUNT` | 2 | process pool size |
| `LW_MAINTENANCE_WINDOW` | (empty = always open) | `HH:MM-HH:MM`, gates `/model/activate` |
| `LW_ROLLBACK_FP_CEILING` | 0.05 | F8 auto-rollback threshold |
| `LW_HOLDOUT_DIR` | ../data/holdout | known-good images for F7/F8 |
| `LW_FLEET_URL` | (empty = no heartbeat) | fleet service base URL, e.g. `http://localhost:9000` |
| `LW_DEVICE_ID` | site-1 | this node's identity in the fleet |
| `LW_PORT` | 8000 | used to build the URL the node reports to the fleet |
| `LW_BUNDLE_SECRET` | (dev default, change it) | shared HMAC key for verifying signed local bundles — see F13 below |

## Run the fleet service

```bash
cd fleet_service
uvicorn app.main:app --port 9000
```

Point one or more edge nodes at it with `LW_FLEET_URL=http://localhost:9000`
(each with a distinct `LW_DEVICE_ID` and `LW_PORT` if running several on one
machine). The node pushes a heartbeat every 10s — endpoints:

- `GET /devices` — fleet view (F10)
- `POST /rollout` `{"version": "v2", "stages": [["site-1"], ["site-2", "site-3"]]}`
  — staged rollout; a stage only proceeds if every device in the prior stage
  activated healthily (F11)
- `GET /rollout/{id}` — status: `in_progress` / `complete` / `halted`
- `GET /drift` — devices whose holdout FP rate has moved >0.15 from their
  own baseline (F12)

If the fleet is unreachable, heartbeats fail silently — `/inspect` on the
edge node has no runtime dependency on the fleet (F13).

## Apply a model update while air-gapped (F13, local bundle)

For a site with no fleet connectivity at all (`LW_FLEET_URL` unset), updates
arrive as a signed bundle carried in on removable media or a one-way gateway
transfer, not pulled over the network:

```bash
cd training
python make_bundle.py --version padim-bottle-v1 --out padim-bottle-v1.bundle

curl -X POST http://<edge-node>/model/bundle -F "bundle=@padim-bottle-v1.bundle"
curl -X POST http://<edge-node>/model/activate
```

The bundle is a zip of `{manifest.json, model.onnx, calibration.json}`; the
manifest carries an HMAC-SHA256 signature over both files' content hashes
(`LW_BUNDLE_SECRET`, shared-secret — not full PKI, matches the plan's
"no production security hardening" non-goal, but still a real integrity
check: a bundle with a forged signature or a tampered file is rejected with
`400` before anything touches disk).

## Run the feeder

```bash
cd feeder
python feeder.py --url http://localhost:8000 --rate 600 --duration 60 --concurrency 20
```

Without `--image-dir`, it generates synthetic byte-blobs — fine against the
stage 1-3 stub inference, but the real PaDiM model can't decode them as
images and every request will fail. Now that a real model is running, point
`--image-dir` at a folder with `good/`/`defect/` subfolders, e.g.
`--image-dir training/datasets/mvtec_bottle/test`.

## F2 acceptance criteria — measure inline vs worker

Run the edge node twice, feed it identically, and record both in
`FINDINGS.md`:

```bash
# terminal 1
set LW_INFERENCE_MODE=inline
uvicorn app.main:app --port 8000
# terminal 2
python feeder/feeder.py --rate 1200 --duration 30 --concurrency 20
```

Then repeat with `LW_INFERENCE_MODE=worker` (or unset it — that's the
default) and compare p50/p95/p99 and throughput.

## Train the real model

```bash
cd training
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install anomalib numpy onnx onnxruntime onnxscript
python download_bottle.py   # ~163 MVTec 'bottle' images, not the 5.27GB full archive
python train_padim.py       # fits in seconds (no gradient descent), exports ONNX + calibration.json
```

Output: `edge_node/models/model.onnx` + `edge_node/models/calibration.json`,
and `data/holdout/` refreshed with real `test/good` images. `edge_node/app/inference.py`
loads both automatically — no edge node code changes needed to retrain.

Both scripts disable TLS certificate verification for their one external
download each (MVTec image mirror; pretrained ResNet18 backbone weights).
This was a workaround for a system-clock/CA-bundle issue in the original
build environment — if you hit `CERTIFICATE_VERIFY_FAILED` errors, check
your system clock first. Otherwise, remove the `ssl._create_default_https_
context` / `verify=False` lines in both files; they're not appropriate to
carry into a normal environment.

**Dataset note:** `download_bottle.py` pulls the MVTec AD "bottle" category
from a public mirror. MVTec AD is free for non-commercial research use —
see [mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
for the license. This repo doesn't redistribute the images; the script
downloads them fresh.

## Docker: the full multi-site simulation (§3a)

`docker-compose.yml` runs the fleet plus three edge-node "sites," each a
container constrained to `--cpus=2 --memory=2g` (verified via `docker
inspect`, not just requested). This is how stages 6-8 are actually exercised
now — see `FINDINGS.md`.

```bash
docker compose build
docker compose up -d
```

Each site publishes its port to the host (`site1`→`8001`, `site2`→`8002`,
`site3`→`8003`), fleet stays on `9000`. Containers reach each other by
service name (`http://site1:8000`, `http://fleet:9000`) — that's what
`LW_ADVERTISE_HOST` is for; it's what a node tells the fleet to reach it at,
separate from what it binds to locally.

Try it:

```bash
curl http://localhost:9000/devices                          # all 3 sites registered
curl -X POST http://localhost:9000/rollout -H "Content-Type: application/json" ^
  -d "{\"version\": \"padim-bottle-v1\", \"stages\": [[\"site-1\"], [\"site-2\", \"site-3\"]]}"
```

Both `edge_node/models/` and `data/holdout/` are mounted read-only into
every site container — retraining locally (`training/train_padim.py`)
updates what all three containers serve without a rebuild, just a restart
(`docker compose restart site1 site2 site3`). The read-only mount is also
why `POST /model/bundle` (F13's local-update path, above) was verified
against a standalone local process rather than a container — a genuinely
air-gapped site would mount its model directory read-write for exactly this
reason; `docker-compose.yml`'s mount is `:ro` because these three
containers are meant to be fleet-managed, not air-gapped, by default.

```bash
docker compose down   # stop everything
```

The single-container form (`docker build`/`docker run` inside `edge_node/`)
still works too, for testing one node in isolation without the fleet.

## Operator review UI (F14)

```
GET /review
```

on any edge node — a plain HTML page (no build step, no framework) listing
recent rejects with their thumbnail and score, newest first, with a
"mark wrong" button per card. Overrides show up in `/health` as
`operator_override_rate` — a leading drift signal, per the plan's F12
framing: an operator noticing something's off is often faster than the
next holdout-based heartbeat check.

## FP32 vs INT8 quantization (F15)

```bash
cd training
python quantize_compare.py --version padim-bottle-v1
```

Dynamic-quantizes the given model to INT8 and benchmarks both against the
real holdout + defect sets on this machine — latency and score-separation,
side by side. On this hardware the result was a genuine regression on both
axes (see `FINDINGS.md`): the point of measuring instead of assuming.

## Status

Everything in the plan (F1-F15) is built and exercised with real evidence.
Nothing is left on the build-order list. Remaining open items — a subtler
rollback test, a client-operable fleet mode, real asymmetric bundle
signing — are tracked honestly in `TECH_DEBT.md`, not hidden.

## License

MIT — see `LICENSE`.
