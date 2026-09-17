# TECH_DEBT

Architectural gaps and open decisions surfaced during the build, not yet
designed or implemented. Distinct from `FINDINGS.md` (what's been measured)
and the README's "Next steps" (small, scoped, already-designed work).

## 1. Client-operable fleet service is a real deployment mode, not documented as one

Discovered while discussing Edgemonitor's actual on-prem client base: many
sites have no internet connectivity, so Edgemonitor (the vendor) cannot assume
live access for updates or data collection.

`fleet_service/` has no vendor coupling baked into it — it's a plain
FastAPI+SQLite service that watches whatever edge nodes point
`LW_FLEET_URL` at it. Nothing about it assumes Edgemonitor operates it.
Consequence: **a client's own IT department can run the fleet service
inside their own plant network**, giving them real-time heartbeat, health,
staged rollout, and drift visibility entirely on their own LAN — no
air-gap problem for anything they want to see themselves, because
site-to-fleet traffic never has to leave the client's network.

This reframes who the fleet service is "for": right now the docs/plan
implicitly assume Edgemonitor operates one central fleet across customers.
A client-operated fleet-per-customer (or fleet-per-site) is a cleaner fit
for genuinely air-gapped or connectivity-restricted deployments, and
changes who owns operational decisions like the rollback ceiling and
promotion gates (plan §10 Q5: "who owns the threshold" — a client running
their own fleet plausibly owns it outright).

**Not yet done:**
- Document this as an explicit supported deployment mode (README + maybe
  the build plan itself), not just something that happens to work.
- Consider a docker-compose variant with no Edgemonitor-facing export at all,
  for a fully client-operated deployment.

## 2. No data path back to Edgemonitor from an air-gapped or client-operated fleet

Follow-on from #1. If a client runs their own fleet (or a site is simply
offline), Edgemonitor gets **zero visibility**, not delayed visibility —
there's no mechanism for it at all today. The signed bundle mechanism
(F13) solves the *update* direction (Edgemonitor → site) for offline sites;
nothing symmetric exists for the *data* direction (site/fleet → Edgemonitor).

**Design shape that seems right, not yet built:** don't sync every site
individually — sync at most **one aggregate point per client**: an
opt-in, periodic or manual export *from the client's fleet service*
(not from every edge node) to Edgemonitor, only if/when the client agrees to
share anything. Mirrors the bundle mechanism's shape (signed, offline-
friendly, operator-initiated) but in the reverse direction.

**Also not yet built, smaller piece:** `heartbeat.py` today is pure
fire-and-forget — on failure it just retries in 10s with no local
accumulation of what was missed. Even within a single edge-node-to-fleet
relationship (no client-operated fleet involved), a site with
intermittent connectivity loses every health snapshot sent during an
outage. Store-and-forward (queue snapshots locally, flush when the fleet
becomes reachable again) would fix this regardless of the larger
client-operated-fleet question.

## 3. Model differentiation in F8/F11 relies on a wrong-category model, not a genuinely degraded one

The rollback and rollout-halt demonstrations (`FINDINGS.md` stages 5/6)
use `padim-screw-v1` as the "bad model" — a real model that's simply
trained on the wrong object. That's a strong, honest test of the platform
*mechanism*, but it's the easy case: a wrong-category model fails
obviously and immediately (100% FP rate). The harder, more realistic
failure mode — a model for the *right* object that's subtly worse (e.g.
trained on fewer images, or with a slightly-off threshold calibration) —
hasn't been tested. Worth doing before trusting the rollback ceiling
(`LW_ROLLBACK_FP_CEILING=0.05`) tuned against an easy case to also catch
a hard one.
