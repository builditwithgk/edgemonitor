# FINDINGS

Numbers as they're actually measured on this machine, in build order. Fill in
as each stage's acceptance criteria is exercised.

## Stage 2 — inline vs worker-pool inference (F2)

Machine: laptop, no external load. Stub `score_image()` does `time.sleep(0.015)`
to stand in for a ~15ms CPU forward pass. `LW_WORKER_COUNT=2` (default).

| Mode | Target rate (req/min) | Concurrency | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/min) |
|---|---|---|---|---|---|---|---|
| inline | 1200 | 20 | 328 | 22.67 | 25.41 | 27.92 | 982.5 |
| worker | 1200 | 20 | 351 | 23.63 | 25.29 | 91.98 | 1050.5 |
| inline | 6000 | 50 | 901 | **680.05** | 901.41 | 903.70 | 3277.9 |
| worker | 6000 | 50 | 990 | **23.82** | 26.25 | 111.04 | 3956.9 |

**Takeaway:** at low concurrency the two modes look identical — not enough
in-flight requests to expose the difference. Under real concurrent load
(50 in flight), inline inference blocks the single event loop for the full
15ms of every request, so requests queue behind each other and p50 balloons
to 680ms — a ~28x regression vs. the worker-pool p50 of 23.8ms, which stays
flat because two OS processes actually run inference in parallel while the
event loop keeps accepting new connections. This is the collapse the plan
predicted (F2): the bug is invisible until you generate genuine concurrency.

## Stage 3 — feeder sustained load (F4)

| Target rate (parts/min) | Duration (s) | p50 (ms) | p95 (ms) | p99 (ms) | Failures |
|---|---|---|---|---|---|
| 1000 | 60 | 23.59 | 25.36 | 37.91 | 0 |

896 requests over 60s (~896/min observed, close to the 1000/min target — the
feeder's own async scheduling overhead accounts for the gap). API stayed
responsive throughout; `/health` immediately after showed `status: ok` with
no backlog (`seconds_since_last_success` ~6s, i.e. no queue pileup).

## Stage 1 — real model: PaDiM on MVTec AD 'bottle' (deferred piece, now done)

Trained via `training/train_padim.py`: anomalib PaDiM, ResNet18 backbone
(`layer1`, `layer2`), CPU only, batch size 1. 80 subsampled `train/good`
images (of 209 available), fit in ~5s (no gradient descent — PaDiM is
forward-pass feature extraction + per-patch Gaussian fitting). Exported to
ONNX; the exported graph outputs the raw score only, normalized to [0, 1]
in `edge_node/app/inference.py` using calibration stats (`image_min`,
`image_max`) saved at training time — see the code comment there for why
(anomalib's own baked-in ONNX normalization didn't reproduce the live
model's calibration; verified with a side-by-side comparison).

Score separation on the untouched 20-image `test/good` + 63-image
`test/defect` holdout, via the actual live `/inspect` endpoint:

| Set | n | min | max |
|---|---|---|---|
| good | 20 | 0.010 | 0.142 (sampled: 8/20) |
| defect | 63 | 0.328 | 0.576 (sampled: 8/63) |

Clean gap between ~0.14 and ~0.33 on the sampled images — no overlap.

## Threshold vs false-positive rate (F3)

Measured against the real 20-image holdout set (`data/holdout/`, real
MVTec bottle `test/good` images) via `GET/PUT /config/threshold` on the
live edge node running the real PaDiM model:

| Threshold | Holdout FP rate |
|---|---|
| 0.10 | 0.25 |
| 0.15 | 0.05 |
| 0.20 | 0.00 |
| 0.25 | 0.00 |
| 0.30 | 0.00 |
| 0.40 | 0.00 |
| 0.50 | 0.00 |

This is a real, non-monotonic-looking-but-actually-just-flat curve once
past 0.20 — exactly the "threshold, not the model, dominates false-positive
rate" point from the plan's §1: on this holdout set nearly any threshold
above 0.2 gives zero false positives, but the interesting behaviour is at
the boundary (0.10-0.15) where it changes fast. Running default: 0.25.

## Stage 4 — hot model swap (F6)

Feeder run at 1200 req/min, concurrency 15, for 20s. At the 8s mark (feeder
still running), staged `stub-v2` via `POST /model` then activated it via
`POST /model/activate`.

- Stage+activate round trip: ~732ms (two sequential HTTP calls, not the
  swap itself — the in-process version flip is a dict pointer swap, effectively
  instant; the 732ms is almost entirely two HTTP request/response cycles)
- Requests dropped during swap: 0
- Requests failed during swap: 0 (332/332 succeeded)
- `/model/activate` response confirmed `active_version: stub-v2`,
  `previous_version: stub-v1`, `rolled_back: false`
- No restart occurred — same running process, same event loop, feeder never
  saw a connection error

## Stage 5 — automatic rollback (F8)

Redone against real models, per the plan's stage 5 acceptance criteria
("one trained on a different MVTec category"): trained a second PaDiM model
on MVTec AD's `screw` category (`training/train_padim.py --category screw
--version-name padim-screw-v1`, same recipe as bottle — 40 train images).
`padim-bottle-v1` was active and healthy (0% holdout FP). Staged and
activated `padim-screw-v1` on the live bottle line via the real API
(`POST /model`, `POST /model/activate`).

- Bad model used: `padim-screw-v1` — a real model, just trained on the wrong
  object, evaluated against real bottle holdout images
- Holdout FP rate that triggered rollback: **1.000** (20/20 bottle holdout
  images scored as anomalous by the screw-trained model — its feature
  statistics have nothing to do with bottles, so everything looks anomalous)
- Time from activation call to reversion decision: ~1666ms (synchronous;
  higher than the stub-era ~518ms because this cold-loads a second ONNX
  Runtime session + evaluates 20 real images through it, vs. the stub's
  instant hash-based scoring)
- Result: `active_version` reverted to `padim-bottle-v1`, `rolled_back: true`,
  and `/health.rollback_events` shows `"holdout false-positive rate 1.000
  exceeded ceiling 0.050"`. No human action taken.
- This supersedes the earlier stub-based version of this test (labeled
  `bad-model-v1`, testing the rollback *mechanism* only, since the stub
  scores by byte hash and doesn't vary by model version) — this run is a
  genuine model-quality regression, caught automatically.

## Stage 6 — staged rollout (F11)

Redone as real Docker containers via `docker-compose.yml` (§3a: `--cpus=2
--memory=2g` per site, confirmed via `docker inspect` — `NanoCpus:
2000000000`, `Memory: 2147483648`), matching the plan's stage 6 acceptance
criteria exactly: three nodes, promote to node 1, check health, promote to
2 and 3, and confirm a failure at node 1 halts before it reaches the others.

- **Setup:** `fleet` + `site1`/`site2`/`site3` containers, each site running
  the real `padim-bottle-v1` model, each independently registering with the
  fleet over the Docker network (`http://site1:8000` etc., not `localhost` —
  required a small fix, `LW_ADVERTISE_HOST`, since each container needs its
  own hostname reported to the fleet).
- **Clean 2-stage rollout:** `padim-bottle-v1` (a no-op re-promotion) to
  `[[site-1], [site-2, site-3]]` → stage 1 completed and passed health,
  stage 2 promoted the remaining two → `status: complete`,
  "promoted to all 3 devices", ~33s total (dominated by each node
  evaluating 20 real holdout images through the real ONNX model per
  activation, over the Docker network).
- **Real failure halts the rollout before reaching later nodes:** same
  2-stage rollout, this time with `padim-screw-v1` (the wrong-category
  model from stage 5) as the version → `site-1` activated it, evaluated
  100% holdout FP rate, auto-rolled-back per F8 — and the rollout halted
  at `status: halted` with `site-1`'s failure in the detail.
  **`site-2` and `site-3` were never touched** — confirmed via `GET /devices`
  showing both still on `padim-bottle-v1` with an unchanged `previous_version`,
  because stage 2 of the rollout never ran. This is the actual acceptance
  criteria from the plan, not a proxy for it.

## Stage 7 — drift (F12)

Redone per the plan's stage 7 acceptance criteria exactly: "feeding images
with altered brightness or contrast shifts the score distribution."

- Perturbation applied: the 20 real bottle holdout images were altered
  in-place (Pillow `ImageEnhance` — brightness x1.8, contrast x1.6, standing
  in for a plant's lighting drifting over time per the plan's §2 constraint
  table) while `padim-bottle-v1` stayed active, unchanged.
- Score distribution shift observed: holdout FP rate at threshold 0.25 moved
  from 0.0 (baseline, captured at the first heartbeat) to **0.35** — purely
  from the lighting change, no model or threshold change. The next 10s
  heartbeat picked this up and `db.upsert_device` flagged `drifting: 1`
  (delta 0.35 > the 0.15 drift threshold); confirmed in `GET /drift`.
- Images restored to original immediately after: FP rate returned to 0.0.
- This is the real version of the earlier stub-era test (which faked drift
  by changing `/config/threshold` directly — a valid mechanism check, but
  not the actual failure mode the plan describes). This run demonstrates
  the harder problem in the plan's own words (§10 Q4): the model didn't get
  worse, the *input distribution* did, and the platform caught it without
  knowing which one had changed.

## Stage 8 — air-gap (F13)

Redone as a real Docker container isolation test, matching the plan's stage
8 wording ("with the fleet service stopped, a node continues inspecting
normally"): `docker stop edgemonitor-fleet-1` (the actual container, not just
killing a local process), then fed `site1` directly on its published port.

- 71/71 requests succeeded (real bottle test images, real ONNX inference,
  concurrency 10) with the fleet container fully stopped. `/health` still
  reported `status: ok`.
- p50 670ms / p99 6085ms — real numbers under the §3a container constraint
  (2 CPUs, 2GB RAM) plus real PaDiM inference at concurrency 10; much higher
  than the local-process F2/F4 numbers earlier in this file, which used the
  15ms stub. Not a regression — a legitimately more expensive model running
  under a legitimately tighter resource limit.
- Heartbeat POSTs fail silently inside a try/except (`heartbeat.py:_send_once`)
  and never touch the `/inspect` code path — confirmed again here, now with
  container-to-container networking actually failing (connection refused)
  rather than just a local process being down.
- Fleet container restarted afterward; `site1`/`site2`/`site3` resumed
  heartbeating normally.

**F13's other half — applying a signed local bundle while air-gapped —
is now built** (`edge_node/app/bundle.py`, `training/make_bundle.py`):
a bundle is a zip of `{manifest.json, model.onnx, calibration.json}`, where
the manifest carries an HMAC-SHA256 signature over the content hashes
(shared-secret, not a full PKI — matches the plan's non-goal of skipping
production security hardening, but still proves the integrity/origin check
and the offline update path for real). `POST /model/bundle` verifies before
writing anything to disk.

- Started a standalone edge node with `LW_FLEET_URL` empty (fully
  air-gapped, not just fleet-unreachable) running only `padim-bottle-v1`.
- Built `padim-bottle-v2.bundle` on a separate machine role (the training
  side) via `make_bundle.py`; the v2 model files did not exist on the node
  beforehand — deleted them first specifically so the bundle transfer would
  prove something rather than overwrite identical files.
- `POST /model/bundle` (multipart upload, no fleet involved) → verified
  signature + both content hashes → staged `padim-bottle-v2`.
- `POST /model/activate` → `active_version: padim-bottle-v2`, 0% holdout FP.
- `POST /inspect` against a real image → `model_version: padim-bottle-v2`,
  real score, `verdict: pass` — the whole path works with zero network
  calls to anything except the one local multipart upload.
- **Tamper check:** built a bundle with a forged signature (same content,
  wrong signature) → `POST /model/bundle` returned `400 signature
  verification failed`, nothing written to disk. Confirms the endpoint
  actually verifies rather than trusting the manifest's own claims.

## Stage F14 — operator review UI (Tier 4, optional)

Minimal page at `GET /review` (`edge_node/app/main.py` + `reviews.py`):
every `/inspect` call that results in `reject` is stored (160x160 JPEG
thumbnail, score, model version, timestamp — bounded ring buffer of 200,
not a full-frame archive) and shown newest-first with a "mark wrong"
button. Overrides feed `operator_override_rate` in `/health`, per the
plan's F12 framing of override rate as a drift signal.

Verified for real, not just unit-tested: fed 37 real bottle images
(defect-biased) into a standalone node, opened `/review` in an actual
browser tab, read the rendered page (28 real reject cards with real scores,
e.g. one at 0.926), clicked "mark wrong" on the first card via the browser's
own click handling (not a direct API call), and confirmed `/health` picked
it up: `operator_override_rate` went from `null` to `0.0357` (1/28).

## Stage F15 — FP32 vs INT8 quantization comparison (Tier 4, optional)

`training/quantize_compare.py`: dynamic INT8 quantization (`onnxruntime.
quantization.quantize_dynamic`, weights-only, no calibration set needed) of
`padim-bottle-v1.onnx`, then FP32 vs INT8 compared on the same real holdout
(20 good) + defect (63 image) sets, 5 repeats each, on this machine.

| | FP32 | INT8 | Change |
|---|---|---|---|
| File size | 168.2 MB | 43.4 MB | **-74.2%** |
| p50 latency | 34.86 ms | **954.11 ms** | **~27x slower** |
| Good score range | 0.0101 – 0.1791 | 0.4105 – 0.4451 | range shifted up, **collapsed to a narrow band** |
| Defect score range | 0.1494 – 1.0 | 0.4078 – 1.0 | lower bound now overlaps good's range |
| Good score mean | 0.0756 | 0.4350 | |
| Defect score mean | 0.4277 | 0.4849 | |

**Both real numbers, and both bad.** Two independent regressions, not one:

1. **Slower, not faster.** Dynamic quantization only converts MatMul/Gemm
   weights to INT8 (a warning during quantization noted an unsupported
   tensor in the anomaly-map blur op, so this graph didn't fully quantize
   anyway) — this PaDiM/ResNet18 backbone is conv-heavy, and this CPU/
   onnxruntime build has no fast vectorized INT8 conv kernel path for it.
   The result is Q/DQ (quantize/dequantize) conversion overhead on every
   op boundary with none of the expected speedup: ~27x *slower*, not
   faster. Dynamic quantization is not free, and its benefit is highly
   graph- and hardware-dependent — the plan's F15 exists precisely so this
   gets measured instead of assumed.
2. **Worse separation, not just worse latency.** The good/defect score
   ranges (after the same min-max normalization) now overlap almost
   entirely — INT8's good-image mean (0.435) is *higher* than FP32's
   defect-image mean (0.428). At the running threshold (0.25), the INT8
   model would reject nearly everything, good parts included. Quantization
   error accumulated through PaDiM's per-patch Gaussian distance
   calculation is enough to destroy the anomaly score's meaning, not just
   shift it.

**Conclusion for this model/hardware combination: do not ship the INT8
variant.** Smaller file size is not, on its own, a reason to quantize —
both the latency and accuracy claims have to be measured, and here neither
held up. (`edge_node/models/padim-bottle-v1-int8.onnx` is kept on disk as
evidence; it is not referenced by the running edge node.)
