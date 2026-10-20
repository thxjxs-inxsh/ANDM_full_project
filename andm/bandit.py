"""
bandit.py
---------
Implements SRS Section 2.2 Item 4: "Edge-Native Dual-Loop Learning" via an
on-device Multi-Armed Bandit that adapts routing using implicit feedback
(dismiss times, tap rates).

We use Beta-Bernoulli Thompson Sampling, one bandit "arm pair" per
(category, sender_bucket) key. Two arms per key: DELIVER vs DIGEST.
(HOLD is reserved for the hard safety/timeout bypass path and is never
a bandit-selected arm, matching SRS 4.1.3 REQ-NLP-3 / Section 5.2.)

Reward signal:
  - Notification tapped/opened while DELIVERed -> reward 1 for DELIVER arm
  - Notification dismissed without opening while DELIVERed -> reward 0
  - Notification opened later from the Smart Digest card -> reward 1 for
    DIGEST arm (validates the hold-and-batch decision)
  - Digest item never opened -> reward 0 for DIGEST arm

Persistence uses the same SQLite file as graph_engine so the whole
middleware state survives a restart, matching SRS 3.3.3 local storage
requirements (encryption/SQLCipher is a production concern out of scope
for this web simulator -- see README).
"""
from __future__ import annotations

import sqlite3
import random
from pathlib import Path

DB_FILE = Path(__file__).parent / "andm_graph.db"

ARMS = ("DELIVER", "DIGEST")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bandit_arms (
                arm_key TEXT NOT NULL,
                arm TEXT NOT NULL,
                alpha REAL NOT NULL DEFAULT 1.0,
                beta REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (arm_key, arm)
            )
            """
        )
        conn.commit()


def _get_or_init(conn: sqlite3.Connection, arm_key: str, arm: str) -> tuple[float, float]:
    row = conn.execute(
        "SELECT alpha, beta FROM bandit_arms WHERE arm_key=? AND arm=?",
        (arm_key, arm),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO bandit_arms (arm_key, arm, alpha, beta) VALUES (?,?,1.0,1.0)",
            (arm_key, arm),
        )
        conn.commit()
        return 1.0, 1.0
    return row


def sample_action(arm_key: str) -> tuple[str, dict]:
    """Thompson-sample the best arm for this key. Returns (arm_name, debug_scores)."""
    with _connect() as conn:
        scores = {}
        for arm in ARMS:
            alpha, beta = _get_or_init(conn, arm_key, arm)
            scores[arm] = random.betavariate(alpha, beta)
    best = max(scores, key=scores.get)
    return best, scores


def expected_value(arm_key: str, arm: str) -> float:
    """Posterior mean (alpha / (alpha+beta)) -- used as a continuous
    'bandit confidence' signal to blend into the routing score, separate
    from the discrete sampled action."""
    with _connect() as conn:
        alpha, beta = _get_or_init(conn, arm_key, arm)
    return alpha / (alpha + beta)


def preference(arm_key: str) -> tuple[float, float, float]:
    """Learned preference for delivering NOW vs. batching into the digest.
    Uses BOTH arms' posterior means, so digest feedback influences routing.

        pref = 0.5 + 0.5 * (EV_deliver - EV_digest)      -> in [0, 1]

    0.5 = no preference (fresh key), >0.5 = favours DELIVER, <0.5 = favours DIGEST.
    Returns (pref, ev_deliver, ev_digest).
    """
    ev_deliver = expected_value(arm_key, "DELIVER")
    ev_digest = expected_value(arm_key, "DIGEST")
    pref = 0.5 + 0.5 * (ev_deliver - ev_digest)
    return max(0.0, min(1.0, pref)), ev_deliver, ev_digest


def update(arm_key: str, arm: str, reward: float) -> None:
    """reward in [0, 1] (Bernoulli-style; 1 = positive engagement)."""
    reward = max(0.0, min(1.0, reward))
    with _connect() as conn:
        alpha, beta = _get_or_init(conn, arm_key, arm)
        alpha += reward
        beta += (1.0 - reward)
        conn.execute(
            "UPDATE bandit_arms SET alpha=?, beta=? WHERE arm_key=? AND arm=?",
            (alpha, beta, arm_key, arm),
        )
        conn.commit()


def all_arms(limit: int = 100) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT arm_key, arm, alpha, beta FROM bandit_arms LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "arm_key": k,
            "arm": a,
            "alpha": round(al, 2),
            "beta": round(be, 2),
            "expected_value": round(al / (al + be), 3),
        }
        for (k, a, al, be) in rows
    ]


init_db()