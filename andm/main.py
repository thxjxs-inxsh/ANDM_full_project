"""
main.py
-------
Adaptive Notification Delivery Middleware (ANDM) -- Web Simulation Layer.

This is the full pipeline described across SRS Sections 2.2, 4.1, 4.2, 5.1,
5.2 and 5.5, wired together for the web-based architectural simulator
described in SRS Section 1.5. It builds on the NLP interception you already
had working (cat.py) and adds the remaining functional blocks:

  1. Notification Payload Interception & Text Extraction  [you had this]
  2. Telemetry Stream Aggregation (LightGBM GBDT)          [telemetry_model.py]
  3. Heterogeneous Graph Structural Mapping (simplified)   [graph_engine.py]
  4. Edge-Native Dual-Loop Learning (Multi-Armed Bandit)    [bandit.py]
  5. Real-Time Action Egress Routing                       [this file]

Safety-critical behavior (SRS 3.1.1, 5.2, 5.5):
  - Emergency/critical channels ALWAYS bypass the ML stack -> DELIVER.
  - No error dialogs are ever shown to the end user; failures fail open
    (deliver) and are logged, per SRS 3.1.1 "Silent Error Reporting" +
    5.2 Safety Requirements.

NOTE: the SRS 15 ms REQ-NLP-3 timeout fail-safe has been removed for this
CPU simulator; the pipeline always completes (~1 s is acceptable).
"""
from __future__ import annotations

import time
import datetime
from pathlib import Path
from typing import Optional, Literal

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import graph_engine as ge
import bandit as bd
from telemetry_model import get_context_score

# --------------------------------------------------------------------------
# Config (SRS 2.5 Design and Implementation Constraints)
# --------------------------------------------------------------------------
TOTAL_LATENCY_BUDGET_MS = 1000.0  # simulator: up to ~1s is fine
EMERGENCY_CHANNELS = {
    "emergency_alerts", "amber_alert", "severe_weather", "sos",
    "phone_call", "system_alarm", "security_verification",
}

ROUTING_WEIGHTS = {"telemetry": 0.40, "graph": 0.30, "bandit": 0.30}
DELIVER_THRESHOLD = 0.60
DIGEST_THRESHOLD = 0.30
BANDIT_TIEBREAK_MARGIN = 0.05   # near the DELIVER threshold, Thompson sampling decides

app = FastAPI(title="ANDM Notification Processor (Full Pipeline)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --------------------------------------------------------------------------
# 1. NLP interception model (unchanged from cat.py, kept intact)
# --------------------------------------------------------------------------
import classifier as clf  # hardened shared classifier (rules + kNN + safe fallbacks)
centroids = clf.centroids


def classify_intent(text: str):
    r = clf.classify(text)
    return r["category"], r["confidence"], r["vector"], r


# --------------------------------------------------------------------------
# In-memory simulated telemetry state (SRS 3.2: passive, non-wake polling).
# The dashboard writes here via POST /telemetry; /intercept-notification
# reads it the way the real middleware would read cached sensor registers.
# --------------------------------------------------------------------------
current_telemetry = {
    "speed_mps": 0.0,
    "imu_magnitude": 0.0,
    "is_charging": False,
    "is_unmetered_wifi": False,
    "calendar_busy": False,
}


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class NotificationPayload(BaseModel):
    title: str
    body: str


class FCMMessage(BaseModel):
    token: Optional[str] = None
    notification: NotificationPayload
    android_channel_id: Optional[str] = None
    sender_id: Optional[str] = "unknown_sender"


class FCMRoot(BaseModel):
    message: FCMMessage


class TelemetryUpdate(BaseModel):
    speed_mps: Optional[float] = None
    imu_magnitude: Optional[float] = None
    is_charging: Optional[bool] = None
    is_unmetered_wifi: Optional[bool] = None
    calendar_busy: Optional[bool] = None


class FeedbackPayload(BaseModel):
    sender_id: str
    category: str
    action_taken: Literal["DELIVER", "DIGEST", "HOLD"]
    signal: Literal["opened", "dismissed", "digest_opened", "digest_ignored"]


# --------------------------------------------------------------------------
# Routing egress (SRS 2.2 Item 5 / Section 4)
# --------------------------------------------------------------------------
def route_notification(category: str, confidence: float, sender_id: str, channel: Optional[str],
                        telemetry: dict) -> dict:
    """Returns a dict describing the final routing decision and the
    breakdown of every signal that fed into it, for dashboard transparency.
    """
    # --- Safety / business-rule bypass (SRS 5.2, 5.5) -----------------
    if channel and channel.lower() in EMERGENCY_CHANNELS:
        return {
            "action": "DELIVER",
            "reason": "emergency_channel_bypass",
            "breakdown": None,
        }

    if category == "Immediate":
        return {"action": "DELIVER", "reason": "nlp_class_immediate", "breakdown": None}

    if category == "Ignore":
        return {"action": "DIGEST", "reason": "nlp_class_ignore", "breakdown": None}

    # --- Adaptive class: fuse telemetry + graph + bandit --------------
    hour_of_day = datetime.datetime.now().hour
    telemetry_score = get_context_score(
        speed_mps=telemetry["speed_mps"],
        imu_magnitude=telemetry["imu_magnitude"],
        is_charging=telemetry["is_charging"],
        is_unmetered_wifi=telemetry["is_unmetered_wifi"],
        calendar_busy=telemetry["calendar_busy"],
        hour_of_day=hour_of_day,
    )
    graph_affinity = ge.get_sender_affinity(sender_id)

    arm_key = f"{category}::{sender_id}"
    # Thompson sample: a random draw from each arm's Beta posterior (exploration).
    bandit_action, bandit_scores = bd.sample_action(arm_key)
    # Posterior-mean preference built from BOTH arms (DELIVER vs DIGEST), so
    # digest feedback (digest_opened / digest_ignored) now changes routing too.
    bandit_pref, ev_deliver, ev_digest = bd.preference(arm_key)

    final_score = (
        ROUTING_WEIGHTS["telemetry"] * telemetry_score
        + ROUTING_WEIGHTS["graph"] * graph_affinity
        + ROUTING_WEIGHTS["bandit"] * bandit_pref
    )

    if final_score >= DELIVER_THRESHOLD:
        action = "DELIVER"
    elif final_score >= DIGEST_THRESHOLD:
        action = "DIGEST"
    else:
        action = "HOLD"
    decided_by = "score_threshold"

    # Bandit tiebreak: when the fused score is borderline around the DELIVER
    # threshold, let the Thompson-sampled arm pick DELIVER vs DIGEST. This is
    # where the bandit explores and where it actually changes the outcome.
    if abs(final_score - DELIVER_THRESHOLD) <= BANDIT_TIEBREAK_MARGIN:
        action = bandit_action          # always "DELIVER" or "DIGEST"
        decided_by = "bandit_thompson_tiebreak"

    return {
        "action": action,
        "reason": "adaptive_fusion",
        "breakdown": {
            "telemetry_score": round(telemetry_score, 3),
            "graph_affinity": round(graph_affinity, 3),
            "bandit_expected_value": round(bandit_pref, 3),
            "bandit_deliver_ev": round(ev_deliver, 3),
            "bandit_digest_ev": round(ev_digest, 3),
            "bandit_thompson_sample": bandit_action,
            "bandit_thompson_scores": {k: round(v, 3) for k, v in bandit_scores.items()},
            "decided_by": decided_by,
            "final_score": round(final_score, 3),
            "arm_key": arm_key,
        },
    }


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.post("/intercept-notification")
def intercept_notification(payload: FCMRoot):
    start_time = time.time()
    try:
        notif = payload.message.notification
        text = ". ".join(p.strip() for p in (notif.title, notif.body) if p and p.strip())
        sender_id = payload.message.sender_id or "unknown_sender"
        channel = payload.message.android_channel_id

        # --- Emergency bypass short-circuits BEFORE the ML stack runs,
        # exactly as SRS 3.1.1 "Fail-Safe Delivery" and 5.2 require.
        if channel and channel.lower() in EMERGENCY_CHANNELS:
            routing = route_notification("Immediate", 1.0, sender_id, channel, current_telemetry)
            latency_ms = (time.time() - start_time) * 1000
            return {
                "status": "success",
                "classification": "Immediate",
                "confidence_score": 1.0,
                "routing": routing,
                "latency_ms": round(latency_ms, 2),
                "budget_ok": latency_ms <= TOTAL_LATENCY_BUDGET_MS,
                "bypassed_ml": True,
            }

        category, confidence, vector, detail = classify_intent(text)

        routing = route_notification(category, confidence, sender_id, channel, current_telemetry)

        # Log a "no signal yet" interaction row so the sender shows up in
        # the graph immediately; a real reward is attached later via
        # POST /feedback once the user actually engages (or doesn't).
        ge.record_interaction(sender_id, category, routing["action"], None)

        latency_ms = (time.time() - start_time) * 1000
        return {
            "status": "success",
            "classification": category,
            "confidence_score": round(confidence, 4),
            "routing": routing,
            "decision_source": detail["source"],
            "class_scores": detail["class_scores"],
            "vector_dimension": f"1x{vector.shape[1]}" if vector is not None else None,
            "vector_sample": vector.squeeze(0).tolist()[:5] if vector is not None else [],
            "latency_ms": round(latency_ms, 2),
            "budget_ok": latency_ms <= TOTAL_LATENCY_BUDGET_MS,
            "bypassed_ml": False,
        }
    except Exception as e:
        # SRS 3.1.1 "Silent Error Reporting": never show a raw error to the
        # end user -- in the real system this would fail open (deliver) and
        # log silently. The simulator surfaces it via HTTP 500 for debugging.
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/telemetry")
def update_telemetry(update: TelemetryUpdate):
    for field, value in update.model_dump(exclude_none=True).items():
        current_telemetry[field] = value
    return {"status": "ok", "current_telemetry": current_telemetry}


@app.get("/telemetry")
def read_telemetry():
    return current_telemetry


@app.post("/feedback")
def submit_feedback(fb: FeedbackPayload):
    """Closes the Multi-Armed Bandit's dual learning loop (SRS 2.2 Item 4).
    A tap/open is reward=1 for the arm that was actually taken; a dismiss
    or an ignored digest item is reward=0. This also feeds the graph
    engine's sender-affinity score.
    """
    reward_map = {
        "opened": 1.0,
        "dismissed": 0.0,
        "digest_opened": 1.0,
        "digest_ignored": 0.0,
    }
    reward = reward_map[fb.signal]
    arm = "DELIVER" if fb.signal in ("opened", "dismissed") else "DIGEST"

    arm_key = f"{fb.category}::{fb.sender_id}"
    bd.update(arm_key, arm, reward)
    ge.record_interaction(fb.sender_id, fb.category, fb.action_taken, reward)

    return {
        "status": "ok",
        "arm_key": arm_key,
        "arm_updated": arm,
        "reward": reward,
        "new_expected_value": round(bd.expected_value(arm_key, arm), 3),
        "new_sender_affinity": round(ge.get_sender_affinity(fb.sender_id), 3),
    }


@app.get("/state")
def get_state():
    return {
        "senders": ge.list_senders(),
        "bandit_arms": bd.all_arms(),
        "current_telemetry": current_telemetry,
    }


@app.get("/health")
def health():
    return {"status": "ok", "centroid_categories": list(centroids.keys())}


static_dir = Path(__file__).parent / "static"  # put index.html inside ./static/
if static_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="dashboard")


@app.get("/")
def root():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "ANDM backend running. See /docs for the API."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)