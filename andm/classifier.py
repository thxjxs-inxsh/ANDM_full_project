"""
classifier.py -- shared, hardened notification classifier for ANDM.

Pipeline (every notification ALWAYS gets one of the categories in centroids.pt):
  1. Input sanitising   -> empty / garbage text falls back to "Adaptive".
  2. Rule layer         -> unambiguous patterns (OTP, fraud, emergency, promo,
                           receipt...) are decided instantly and deterministically.
  3. Embedding kNN      -> MiniLM vector vs. exemplar matrices; each class is
                           scored by the mean of its TOP_K nearest exemplars
                           (more robust than a single nearest neighbour).
  4. Safety tie-break   -> if "Immediate" is nearly tied with the winner,
                           pick "Immediate" (never silently bury something urgent).
  5. Low-confidence     -> if nothing matches well (novel/odd text), fall back
                           to "Adaptive" so the context/graph/bandit stack
                           decides, instead of a random hard label.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CENTROIDS_FILE = Path(__file__).parent / "centroids.pt"

TOP_K = int(os.getenv("ANDM_TOP_K", "3"))
LOW_CONF = float(os.getenv("ANDM_LOW_CONF", "0.40"))        # below this -> Adaptive
IMMEDIATE_MARGIN = float(os.getenv("ANDM_IMM_MARGIN", "0.0"))
FALLBACK_CLASS = "Adaptive"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

if not CENTROIDS_FILE.exists():
    raise FileNotFoundError(f"{CENTROIDS_FILE.name} not found. Run build.py first.")
centroids = torch.load(CENTROIDS_FILE, weights_only=True)


def get_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad():
        out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        pooled = (out.last_hidden_state * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        return F.normalize(pooled, p=2, dim=1)


# ---------------------------------------------------------------- rule layer
_I = re.I
IMMEDIATE_RULES = [re.compile(p, _I) for p in [
    r"\b(otp|one[- ]time (pass)?code|verification code|security code|2fa|passcode|login code)\b",
    r"\b(unauthori[sz]ed|suspicious|fraud(ulent)?|unrecogni[sz]ed)\b",
    r"\b(new|unusual) (login|sign-?in|device)\b",
    r"\b(card|payment|transaction|transfer)\b.{0,25}\b(declined|failed|blocked|frozen)\b",
    r"\bpassword\b.{0,20}\b(changed|reset)\b",
    r"\b(emergency|evacuat\w+|earthquake|tsunami|tornado|amber alert|wildfire|take shelter|"
    r"take cover|carbon monoxide|smoke detected|fire alarm|intruder|gas leak)\b",
    r"\b(driver|cab|taxi|uber|lyft)\b.{0,30}\b(arrived|arriving|waiting outside|at your door|is here)\b",
    r"\bincoming call\b",
]]
IGNORE_RULES = [re.compile(p, _I) for p in [
    r"\b\d{1,3}\s?% off\b",
    r"\b(sale|promo(tion(al)?)?|coupon|discount|cashback|voucher|limited[- ]time offer|bogo)\b",
    r"\b(newsletter|unsubscribe|weekly digest|daily digest)\b",
    r"\b(receipt|invoice)\b",
    r"\b(rate (your|us|the)|leave a (review|rating)|how was your|take (a|our) survey)\b",
    r"\b(new followers?|streak|trending|recommended for you|you might like|new arrivals)\b",
    r"\b(monthly|yearly|annual|weekly) (statement|summary|report)\b.{0,40}\b(ready|available)\b",
]]


def _rule_match(text: str):
    for rx in IMMEDIATE_RULES:
        if rx.search(text):
            return "Immediate", rx.pattern
    for rx in IGNORE_RULES:
        if rx.search(text):
            return "Ignore", rx.pattern
    return None, None


def classify(text: str) -> dict:
    """Always returns a dict with a valid category."""
    text = re.sub(r"\s+", " ", (text or "")).strip(" .")
    if len(text) < 2:
        return {"category": FALLBACK_CLASS, "confidence": 0.0, "source": "empty_input_fallback",
                "class_scores": {}, "vector": None}

    vector = get_embedding(text)

    cat, pattern = _rule_match(text)
    if cat:
        return {"category": cat, "confidence": 1.0, "source": "rule",
                "rule": pattern, "class_scores": {}, "vector": vector}

    scores = {}
    for name, matrix in centroids.items():
        sims = torch.matmul(matrix, vector.T).squeeze(-1)
        if sims.ndim == 0:
            scores[name] = sims.item()
        else:
            scores[name] = sims.topk(min(TOP_K, sims.numel())).values.mean().item()

    best = max(scores, key=scores.get)
    source = "embedding_knn"

    if "Immediate" in scores and best != "Immediate" and scores[best] - scores["Immediate"] <= IMMEDIATE_MARGIN:
        best, source = "Immediate", "safety_tiebreak_immediate"

    if scores[best] < LOW_CONF:
        best, source = FALLBACK_CLASS, "low_confidence_fallback"

    return {"category": best, "confidence": scores[best] if source != "low_confidence_fallback"
            else max(scores.values()), "source": source,
            "class_scores": {k: round(v, 4) for k, v in scores.items()}, "vector": vector}


# warm-up so the first real request isn't slow
classify("warm up request")