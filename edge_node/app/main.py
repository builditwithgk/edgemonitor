from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from . import heartbeat, reviews
from .bundle import BundleError, verify_and_apply_bundle
from .config import config
from .holdout import evaluate_false_positive_rate
from .metrics import metrics
from .model_store import model_store
from .worker import run_inference, start_pool, stop_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_pool()
    heartbeat.start()
    yield
    heartbeat.stop()
    stop_pool()


app = FastAPI(title="Edgemonitor Edge Node", lifespan=lifespan)


# ---------------------------------------------------------------- F1, F2 --

class InspectResult(BaseModel):
    verdict: str
    score: float
    model_version: str
    latency_ms: float


@app.post("/inspect", response_model=InspectResult)
async def inspect(image: UploadFile = File(...)):
    image_bytes = await image.read()
    model_version = model_store.active.version
    score, latency_ms = await run_inference(image_bytes, model_version)
    threshold = config.get_threshold()
    verdict = "reject" if score > threshold else "pass"
    metrics.record(latency_ms, rejected=(verdict == "reject"))
    if verdict == "reject":
        reviews.record_reject(image_bytes, score, model_version)  # F14
    return InspectResult(
        verdict=verdict, score=score, model_version=model_version, latency_ms=latency_ms
    )


# --------------------------------------------------------------------- F7 --

@app.get("/health")
async def health():
    fp_rate = evaluate_false_positive_rate(model_store.active.version)
    return {
        "status": "ok",
        "device_id": config.device_id,
        "inference_mode": config.inference_mode,
        **model_store.status(),
        "holdout_false_positive_rate": fp_rate,
        "operator_override_rate": reviews.override_rate(),
        **metrics.snapshot(),
    }


# --------------------------------------------------------------------- F3 --

class ThresholdPayload(BaseModel):
    threshold: float


@app.get("/config/threshold")
async def get_threshold():
    threshold = config.get_threshold()
    fp_rate = evaluate_false_positive_rate(model_store.active.version)
    return {"threshold": threshold, "implied_holdout_false_positive_rate": fp_rate}


@app.put("/config/threshold")
async def put_threshold(payload: ThresholdPayload):
    try:
        config.set_threshold(payload.threshold)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    fp_rate = evaluate_false_positive_rate(model_store.active.version)
    return {"threshold": config.get_threshold(), "implied_holdout_false_positive_rate": fp_rate}


# ---------------------------------------------------------- F5, F6, F8, F9 --

class StageModelPayload(BaseModel):
    version: str
    path: str | None = None


@app.post("/model")
async def stage_model(payload: StageModelPayload):
    model_store.stage(payload.version, payload.path)
    return model_store.status()


@app.post("/model/bundle")
async def stage_model_bundle(bundle: UploadFile = File(...)):
    """F13: apply an update from a local signed bundle -- no fleet
    connection required, so this works while air-gapped. Verifies the
    bundle's HMAC signature and content hashes before writing anything,
    then stages it exactly like `POST /model` (still requires a separate
    `POST /model/activate` -- a bundle landing on disk is not the same as
    it being live, same as any other staged model)."""
    bundle_bytes = await bundle.read()
    try:
        version = verify_and_apply_bundle(bundle_bytes)
    except BundleError as e:
        raise HTTPException(status_code=400, detail=str(e))
    model_store.stage(version)
    return {"staged": version, **model_store.status()}


@app.post("/model/activate")
async def activate_model():
    if not config.in_maintenance_window():
        raise HTTPException(
            status_code=409,
            detail=f"outside maintenance window ({config.maintenance_window}); staged model held",
        )
    if model_store.staged is None:
        raise HTTPException(status_code=400, detail="no staged model")

    previous_version = model_store.active.version
    activated = model_store.activate_staged()  # F6: hot swap, no restart, no dropped in-flight work

    fp_rate = evaluate_false_positive_rate(activated.version)
    if fp_rate is not None and fp_rate > config.rollback_fp_ceiling:
        model_store.rollback()
        metrics.record_rollback(
            from_version=activated.version,
            to_version=previous_version,
            reason=f"holdout false-positive rate {fp_rate:.3f} exceeded ceiling "
                   f"{config.rollback_fp_ceiling:.3f}",
        )
        return {
            **model_store.status(),
            "rolled_back": True,
            "holdout_false_positive_rate": fp_rate,
        }

    return {**model_store.status(), "rolled_back": False, "holdout_false_positive_rate": fp_rate}


# --------------------------------------------------------------------- F14 --

@app.get("/reviews/recent")
async def recent_reviews():
    return reviews.list_recent()


@app.get("/reviews/{review_id}/thumbnail")
async def review_thumbnail(review_id: int):
    thumb = reviews.get_thumbnail(review_id)
    if thumb is None:
        raise HTTPException(status_code=404, detail="unknown review id")
    return Response(content=thumb, media_type="image/jpeg")


@app.post("/reviews/{review_id}/override")
async def override_review(review_id: int):
    if not reviews.mark_overridden(review_id):
        raise HTTPException(status_code=404, detail="unknown review id")
    return {"overridden": review_id, "operator_override_rate": reviews.override_rate()}


@app.get("/review", response_class=HTMLResponse)
async def review_page():
    return _REVIEW_PAGE_HTML


_REVIEW_PAGE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Edgemonitor — Operator Review</title>
<style>
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 24px; }
  h1 { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
  .sub { color: #888; font-size: 13px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; }
  .card { background: #1b1b1b; border-radius: 8px; overflow: hidden; border: 1px solid #2a2a2a; }
  .card img { width: 100%; height: 140px; object-fit: cover; display: block; background: #000; }
  .card .body { padding: 8px 10px; }
  .score { font-size: 13px; color: #f88; font-weight: 600; }
  .meta { font-size: 11px; color: #777; margin-top: 2px; }
  button { margin-top: 8px; width: 100%; padding: 6px; border: 1px solid #444; background: #262626;
           color: #eee; border-radius: 4px; cursor: pointer; font-size: 12px; }
  button:hover { background: #333; }
  button.done { background: #2f4a2f; border-color: #3a6a3a; cursor: default; }
  .empty { color: #666; }
</style>
</head>
<body>
<h1>Recent rejects</h1>
<div class="sub">Click "mark wrong" if a rejected part was actually good — this feeds the drift signal.</div>
<div id="grid" class="grid"><div class="empty">Loading…</div></div>
<script>
async function load() {
  const res = await fetch('/reviews/recent');
  const items = await res.json();
  const grid = document.getElementById('grid');
  if (!items.length) { grid.innerHTML = '<div class="empty">No rejects yet.</div>'; return; }
  grid.innerHTML = items.map(item => `
    <div class="card" id="card-${item.id}">
      <img src="/reviews/${item.id}/thumbnail" loading="lazy">
      <div class="body">
        <div class="score">score ${item.score.toFixed(3)}</div>
        <div class="meta">${item.model_version} · ${new Date(item.ts * 1000).toLocaleTimeString()}</div>
        <button onclick="override(${item.id})" ${item.overridden ? 'class="done" disabled' : ''}>
          ${item.overridden ? 'marked wrong' : 'mark wrong'}
        </button>
      </div>
    </div>
  `).join('');
}
async function override(id) {
  await fetch(`/reviews/${id}/override`, { method: 'POST' });
  load();
}
load();
setInterval(load, 8000);
</script>
</body>
</html>"""


# --------------------------------------------------------------- demo UI --

@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    return _DEMO_PAGE_HTML


_DEMO_PAGE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Edgemonitor — Inspect</title>
<style>
  body {
    font-family: system-ui, sans-serif;
    background: #0d0f12;
    color: #eee;
    margin: 0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }
  .card {
    width: 100%;
    max-width: 440px;
    background: #16191d;
    border: 1px solid #262a30;
    border-radius: 14px;
    padding: 32px;
    text-align: center;
  }
  h1 { font-size: 17px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: #8890a0; font-size: 13px; margin: 0 0 26px; }
  .drop {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 2px dashed #333940;
    border-radius: 10px;
    padding: 36px 16px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .drop:hover, .drop.drag { border-color: #4a9d8f; background: #12201d; }
  .drop input { display: none; }
  .drop .icon { font-size: 30px; margin-bottom: 8px; }
  .drop .label { font-size: 13.5px; color: #b0b6c0; }
  .preview { max-width: 100%; max-height: 180px; border-radius: 8px; margin-bottom: 14px; display: none; }
  .result {
    margin-top: 22px;
    padding: 18px;
    border-radius: 10px;
    display: none;
  }
  .result.pass { background: #12241c; border: 1px solid #2f7d5f; }
  .result.reject { background: #2a1414; border: 1px solid #b5453a; }
  .result .verdict { font-size: 20px; font-weight: 700; letter-spacing: 0.02em; }
  .result.pass .verdict { color: #4fd39a; }
  .result.reject .verdict { color: #ef6b60; }
  .result .meta { font-size: 12.5px; color: #8890a0; margin-top: 8px; line-height: 1.6; }
  .again {
    margin-top: 16px;
    background: none;
    border: 1px solid #333940;
    color: #b0b6c0;
    padding: 7px 16px;
    border-radius: 6px;
    font-size: 12.5px;
    cursor: pointer;
    display: none;
  }
  .again:hover { border-color: #4a9d8f; color: #eee; }
  .loading { display: none; color: #8890a0; font-size: 13px; margin-top: 18px; }
  .spinner {
    display: inline-block; width: 14px; height: 14px; border: 2px solid #333940;
    border-top-color: #4a9d8f; border-radius: 50%; animation: spin 0.7s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="card">
  <h1>Edgemonitor — Part Inspection</h1>
  <p class="sub">Upload a photo of a part to inspect it</p>

  <label class="drop" id="drop">
    <input type="file" id="fileInput" accept="image/*">
    <img id="preview" class="preview">
    <div id="dropLabel">
      <div class="icon">📷</div>
      <div class="label">Click to choose an image, or drag one here</div>
    </div>
  </label>

  <div class="loading" id="loading"><span class="spinner"></span>Inspecting...</div>

  <div class="result" id="result">
    <div class="verdict" id="verdict"></div>
    <div class="meta" id="meta"></div>
  </div>

  <button class="again" id="again">Inspect another</button>
</div>

<script>
const drop = document.getElementById('drop');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const dropLabel = document.getElementById('dropLabel');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const verdict = document.getElementById('verdict');
const meta = document.getElementById('meta');
const again = document.getElementById('again');

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) inspect(fileInput.files[0]);
});

['dragover', 'dragleave', 'drop'].forEach(evt => {
  drop.addEventListener(evt, e => {
    e.preventDefault();
    drop.classList.toggle('drag', evt === 'dragover');
  });
});
drop.addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (f) inspect(f);
});

again.addEventListener('click', () => {
  result.style.display = 'none';
  again.style.display = 'none';
  dropLabel.style.display = 'block';
  preview.style.display = 'none';
  fileInput.value = '';
});

let currentController = null;

async function inspect(file) {
  // A second image chosen before the first request finished used to leave
  // the spinner stuck forever -- the first request's success handler would
  // still fire and stomp the UI state the second request had just set.
  // Aborting the stale request (and ignoring its result if it slips
  // through anyway) fixes that regardless of which one finishes first.
  if (currentController) currentController.abort();
  const controller = new AbortController();
  currentController = controller;

  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  dropLabel.style.display = 'none';
  result.style.display = 'none';
  again.style.display = 'none';
  loading.style.display = 'block';

  const form = new FormData();
  form.append('image', file);

  try {
    const res = await fetch('/inspect', { method: 'POST', body: form, signal: controller.signal });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    const data = await res.json();
    if (controller.signal.aborted) return;
    result.className = 'result ' + (data.verdict === 'pass' ? 'pass' : 'reject');
    verdict.textContent = data.verdict === 'pass' ? '✓ PASS' : '✗ REJECT';
    meta.textContent = `anomaly score ${Number(data.score).toFixed(3)}  ·  ${Number(data.latency_ms).toFixed(0)}ms  ·  ${data.model_version}`;
    result.style.display = 'block';
    again.style.display = 'inline-block';
  } catch (e) {
    if (e.name === 'AbortError') return;  // superseded by a newer selection, not a real error
    result.className = 'result reject';
    verdict.textContent = 'Error';
    meta.textContent = e.message || 'Could not reach the inspection service.';
    result.style.display = 'block';
    again.style.display = 'inline-block';
  } finally {
    if (currentController === controller) {
      loading.style.display = 'none';
      currentController = null;
    }
  }
}
</script>
</body>
</html>"""
