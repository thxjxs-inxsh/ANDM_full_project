# ANDM — Adaptive Notification Delivery Middleware (Web Simulation Layer)

> One-page system map, exact decision rule, API contract and known gaps:
> **[SYSTEM.md](SYSTEM.md)**. This README is the longer narrative version.

This extends the payload-interception prototype you already had (`cat.py` /
`build.py`) into the full pipeline described in the SRS (24BAI1482), built
as the web-based architectural simulator specified in **SRS §1.5**.

## What was already working (yours)
- `build.py` — builds `centroids.pt`, a per-category matrix of MiniLM
  sentence embeddings from `rules.json` (Immediate / Adaptive / Ignore).
- `cat.py` — a FastAPI endpoint that embeds an incoming notification and
  classifies it by max cosine similarity against those exemplar matrices.

This is **SRS §4.1, "Notification Ingress Interception and NLP Embedding"**
— feature block 1 of 5. Nothing here needed to change; `main.py` reuses the
exact same embedding + classification logic.

## What's new — the other 4 feature blocks (SRS §2.2)

| # | SRS Feature | File | What it does |
|---|---|---|---|
| 2 | Telemetry Stream Aggregation | `telemetry_model.py` | LightGBM GBDT regressor that turns speed / motion / power / wifi / calendar state into a single "receptivity" score in [0,1]. |
| 3 | Heterogeneous Graph Structural Mapping | `graph_engine.py` | SQLite-persisted sender interaction history with a time-decayed affinity score per sender (simplified stand-in for the full HGNN — see below). |
| 4 | Edge-Native Dual-Loop Learning | `bandit.py` | Beta-Bernoulli Thompson Sampling multi-armed bandit, one DELIVER/DIGEST arm pair per (category, sender). Updated by real user feedback. |
| 5 | Real-Time Action Egress Routing | `main.py` → `route_notification()` | Fuses classification + telemetry + graph + bandit into a final `DELIVER` / `DIGEST` / `HOLD` token, with the safety bypass paths from §3.1.1 / §5.2 / §5.5. |

Plus:
- **`static/index.html`** — the "Visual Diagnostics Dashboard" from SRS
  §1.5: telemetry toggles, a notification composer, a live pipeline
  visualizer, routing-score meters, feedback buttons, and live sender/bandit
  state. Open `http://127.0.0.1:8000/` once the server is running.
- **`/feedback`, `/telemetry`, `/state`, `/health`** endpoints tying it
  together (see `main.py` docstring for the full contract).

## How the routing decision is made

```
incoming notification
   │
   ├─ channel in EMERGENCY_CHANNELS? ──────────────► DELIVER (bypass everything)
   │
   ├─ NLP class == Immediate? ─────────────────────► DELIVER
   ├─ NLP class == Ignore?    ─────────────────────► DIGEST
   │
   └─ NLP class == Adaptive:
         final_score = 0.4·telemetry + 0.3·graph_affinity + 0.3·bandit_EV
         final_score ≥ 0.60  → DELIVER
         final_score ≥ 0.30  → DIGEST
         else                → HOLD
```

For the CPU simulator the **REQ-NLP-3 15 ms timeout fail-safe is not enforced**;
the latency budget is relaxed to `TOTAL_LATENCY_BUDGET_MS = 1000` and reported per
request as `budget_ok` (see `main.py`). Per-request latency on CPU is ~75–120 ms.

## Honest simplifications (read this before you present it)

I did not fake anything to make it "look" done — these are real, working,
swappable components, but they are intentionally simpler than the full SRS
spec because the full versions need data/hardware you don't have yet at
prototype stage. This is *exactly* what SRS §1.3 justifies the Prototype +
Spiral model for ("resolution of open TBD metrics" requires empirical data
first). Be upfront about these if you're asked in a viva:

1. **Graph engine is not a trained HGNN.** A real Heterogeneous Graph
   Neural Network needs a labelled interaction dataset that doesn't exist
   yet (no one has used the middleware). What's implemented is the
   *structural* piece — a persistent sender/notification interaction graph
   with a time-decayed affinity score — which is what a trained HGNN's
   attention output would functionally replace. Swapping in a real DGL
   model later only means replacing `get_sender_affinity()`.

2. **Telemetry GBDT is trained on synthetic data**, generated directly from
   the business rules already fixed in the SRS itself (§5.5, §2.5) — not
   invented from nothing. This is flagged in the SRS's own TBD list as
   needing real empirical telemetry before final tuning (see TBD-03 for the
   closest analogue).

3. **The 15 ms REQ-NLP-3 timeout is not implemented in this simulator.** A
   CPU-run full-precision MiniLM needs ~75–120 ms per notification, so the
   simulator uses a 1000 ms budget and simply reports `budget_ok`. The SRS
   budget assumes an INT8-quantized MobileBERT on an on-device NPU/DSP
   (§1.3 Justification #2); the fail-open behaviour belongs with that build.

4. **SQLCipher/AES-256 encryption (SRS §3.3.3, §5.3) is not implemented.**
   The prototype uses plain SQLite for speed of iteration. This is a
   flagged TODO, not a silent omission — do not present this as
   privacy-compliant without adding it.

5. **Local Differential Privacy for the federated loop (SRS §3.4.2,
   TBD-03)** is out of scope here entirely — no federated sync loop is
   implemented. This middleware only closes the *local* bandit/graph
   learning loop, not the weekly cross-device federated one.

## Running it

```bash
pip install -r requirements.txt
python main.py
# or: uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000/** for the dashboard, or POST directly:

```bash
curl -X POST http://127.0.0.1:8000/intercept-notification \
  -H "Content-Type: application/json" \
  -d '{"message":{"notification":{"title":"John sent you a message","body":"Are we still meeting?"},"sender_id":"john_slack"}}'
```

`centroids.pt` is reused as-is — you don't need to re-run `build.py` unless
you change `rules.json`.

## Suggested next steps (in SRS priority order)
1. Swap the synthetic telemetry training set for real logged sensor data
   once you have a device build to collect from (closes TBD-03's sibling).
2. Replace `graph_engine.get_sender_affinity()` with a trained DGL HGNN
   once you have enough interaction history to train on.
3. Add SQLCipher encryption to `andm_graph.db` before any real user data
   touches it.
4. Quantize the NLP model (INT8 MobileBERT via ONNX Runtime Mobile, per
   SRS §3.3.1) to make the 15ms budget achievable on-device.
