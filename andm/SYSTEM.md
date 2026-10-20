# ANDM — the whole system in one page

Adaptive Notification Delivery Middleware: a **single FastAPI process** that decides,
for every incoming push notification, whether to show it now (**DELIVER**), batch it
(**DIGEST**), or silence it (**HOLD**), `static/index.html` is the dashboard / phone
preview that drives it.

```
FCM payload ─► 1 Intercept ─► 2 NLP class ─► 3 Telemetry ─┐
                                                4 Graph   ├─► 5 Fusion ─► DELIVER / DIGEST / HOLD
                                                5 Bandit  ─┘
```

## The five stages

| # | Stage | Code | Output |
|---|-------|------|--------|
| 1 | Payload interception | `main.py` `/intercept-notification` | `"title. body"` text + sender + channel |
| 2 | NLP classification | `classifier.py` (exemplars in `centroids.pt`) | `Immediate` / `Adaptive` / `Ignore` + confidence |
| 3 | Telemetry fusion | `telemetry_model.py` (LightGBM GBDT) | receptivity score in `[0,1]` |
| 4 | Sender graph | `graph_engine.py` (SQLite) | time-decayed affinity in `[0,1]` |
| 5 | Bandit learning | `bandit.py` (Beta-Bernoulli Thompson) | deliver-vs-digest preference in `[0,1]` |

## Exact decision rule (main.py `route_notification`)

```
channel in EMERGENCY_CHANNELS .............. DELIVER      (ML never runs, bypassed_ml=true)
class == Immediate ........................ DELIVER
class == Ignore ........................... DIGEST
class == Adaptive:
    final = 0.40*telemetry + 0.30*affinity + 0.30*bandit_pref
    final >= 0.60 ......................... DELIVER
    final >= 0.30 ......................... DIGEST
    else .................................. HOLD
    if |final - 0.60| <= 0.05 ............. action = Thompson sample (DELIVER or DIGEST)
```

Adaptive routing is therefore **deliberately stochastic near 0.60** — that band is the
bandit's exploration window, and the dashboard exposes `decided_by` so you can prove it.

## Model layer specifics

- **Classifier**: regex rule layer first (OTP/fraud/emergency → Immediate; promo/receipt/
  survey → Ignore), else MiniLM embeddings vs. per-class exemplar matrices, scored by mean
  of top-3 cosine similarities. Low confidence (`< 0.40`) falls back to `Adaptive` so the
  context stack decides. Unknown/empty text is safe.
- **Telemetry**: GBDT trained at import on synthetic data derived from the SRS business
  rules (speed, IMU, charge, wifi, calendar, hour).
- **Graph**: `senders` + `interactions` tables; affinity = exponentially time-decayed mean
  reward (half-life 3 days); reward only comes from explicit `/feedback`.
- **Bandit**: two arms (`DELIVER`, `DIGEST`) per key `"<category>::<sender>"`;
  `pref = 0.5 + 0.5*(EV_deliver - EV_digest)`.

## HTTP contract

| Endpoint | Direction | Effect |
|---|---|---|
| `POST /intercept-notification` | dashboard → engine | classifies, routes, logs a no-reward interaction row |
| `POST /telemetry` | dashboard → engine | overwrites the in-memory sensor cache |
| `GET /telemetry` | read | current sensor cache |
| `POST /feedback` | dashboard → engine | updates one bandit arm + graph affinity |
| `GET /state` | read | senders, bandit arms, telemetry (dashboard polls every 3 s) |
| `GET /health` | read | liveness + loaded classes |
| `GET /` , `/dashboard/` | browser | the dashboard |

`POST /feedback` accepts `signal ∈ {opened, dismissed, digest_opened, digest_ignored}`
(reward 1 / 0 / 1 / 0).

## Persisted vs. transient state

- `andm_graph.db` — SQLite: `senders`, `interactions`, `bandit_arms` (survives restart).
- `centroids.pt` — exemplar embedding matrices, built by `build.py`; rebuilt by
  `train_classifier.py` after extending `rules.json`.
- telemetry — in-memory only, resets on restart.

## Files at a glance

| File | Role |
|---|---|
| `main.py` | app, routing fusion, endpoints |
| `classifier.py` | rules + embedding kNN + safe fallbacks |
| `telemetry_model.py` | GBDT context score |
| `graph_engine.py` | sender/notification graph + affinity |
| `bandit.py` | Thompson-sampling arms |
| `static/index.html` | dashboard, pipeline visualiser, phone preview |
| `build.py` | build `centroids.pt` from `rules.json` |
| `train_classifier.py` | evaluate → merge dataset into `rules.json` → re-embed |
| `rules.json` / `realistic_notifications.json` | exemplars / evaluation set |
| `cat.py` | original single-file prototype (**also binds :8000 — don't run alongside `main.py`**) |
| `train_middleware.py` | near-duplicate of `train_classifier.py` (redundant) |

## Verification snapshot (2026-10-20, CPU, Python 3.13)

- `93/93` on `realistic_notifications.json` — but all 93 texts are already inside
  `rules.json`, so this is a consistency check, **not** held-out accuracy.
- Rule paths checked: OTP/decline → `Immediate`; "% off" promo → `Ignore`.
- All three egress actions produced by real requests: `DELIVER` (adaptive fusion),
  `DIGEST` (Ignore class), `HOLD` (driving + busy calendar + disliked sender, final 0.204).
- Emergency channel returned `bypassed_ml=true` in 0.02 ms; bad feedback signal → HTTP 422.
- Latency: 74–118 ms per notification (median ~90 ms) against the 1000 ms simulator budget.
- Feedback loop moved bandit EV and sender affinity in the expected direction.

## Run

```bash
py -3.13 -m pip install -r requirements.txt     # torch/transformers already cached
py -3.13 main.py                                # http://127.0.0.1:8000/
```

Dashboard opens in **Live API** mode automatically; it also has a client-only Mock mode.

## Known gaps (intentional, documented)

1. No SQLCipher/AES encryption on `andm_graph.db`; no federated / differential-privacy loop.
2. Telemetry GBDT and graph affinity are stand-ins for trained sensor data and an HGNN.
3. The SRS 15 ms REQ-NLP-3 timeout is **not** enforced in this CPU simulator (1000 ms budget).
4. `graph_engine.prune_old()` (30-day retention) exists but is never called.
5. Dashboard references `#dynamicIsland` / `#islandContent` / `#phoneClock`, which are not
   in the DOM — the island animation silently never runs (all call sites are guarded).
6. Legacy bandit rows seeded with `<cat>_<sender>` keys (e.g. `Adaptive_Slack`) are
   unreachable from `main.py` (`<cat>::<sender>`) and only clutter the arm list.
7. `POST /feedback` credits the arm named by `signal`, not by `action_taken`, so a HOLD
   result marked "opened" credits the DELIVER arm.
8. "Mark Dismissed (-0.5)" in the UI actually sends reward `0.0`, not `-0.5`.
