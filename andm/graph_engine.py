"""
graph_engine.py
----------------
Implements the "Heterogeneous Graph Structural Mapping" feature described in
SRS Section 2.2 (Item 3) and the local persistence requirements of Section
3.3.3 / Section 6.2.

SIMPLIFICATION NOTE (honest, on purpose):
The SRS specifies a full Heterogeneous Graph Neural Network (HGNN) with
attention-weighted node embeddings across Sender / Context / Notification
node types. Training a real HGNN needs a labelled interaction dataset that
does not exist yet for this prototype. Per the SRS's own chosen SDLC model
(Prototype + Spiral, Section 1.3), this module builds the *structural*
skeleton the HGNN will eventually sit on top of:

  - a persistent Sender <-> Notification interaction graph (SQLite,
    stands in for the encrypted on-device SQLite store of Section 3.3.3)
  - a lightweight "attention-like" affinity score computed from a
    time-decayed weighted average of past engagement, which plays the same
    functional role a trained HGNN attention head would play in the final
    routing decision (a scalar affinity per sender in [0, 1])

Swapping in a real DGL-based HGNN later only requires replacing
`get_sender_affinity()` -- every caller only depends on that scalar.
"""
from __future__ import annotations

import sqlite3
import time
import math
from pathlib import Path
from typing import Optional

DB_FILE = Path(__file__).parent / "andm_graph.db"

# Time-decay half-life for engagement history (seconds). 3 days by default.
DECAY_HALF_LIFE_S = 3 * 24 * 3600


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS senders (
                sender_id TEXT PRIMARY KEY,
                first_seen REAL,
                last_seen REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_id TEXT,
                category TEXT,
                action TEXT,
                reward REAL,
                ts REAL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_sender ON interactions(sender_id)"
        )
        conn.commit()


def touch_sender(sender_id: str) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO senders (sender_id, first_seen, last_seen)
            VALUES (?, ?, ?)
            ON CONFLICT(sender_id) DO UPDATE SET last_seen=excluded.last_seen
            """,
            (sender_id, now, now),
        )
        conn.commit()


def record_interaction(sender_id: str, category: str, action: str, reward: Optional[float]) -> None:
    """reward: 1.0 = positive engagement (tapped/opened), 0.0 = dismissed
    without opening, None = no signal yet (e.g. delivered, no feedback)."""
    touch_sender(sender_id)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO interactions (sender_id, category, action, reward, ts) VALUES (?,?,?,?,?)",
            (sender_id, category, action, reward, time.time()),
        )
        conn.commit()


def get_sender_affinity(sender_id: str, default: float = 0.5) -> float:
    """Time-decayed weighted average of past reward signals for this sender.
    Returns a scalar in [0, 1]. New/unknown senders get a neutral `default`.
    This is the stand-in "attention score" described above.
    """
    now = time.time()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT reward, ts FROM interactions WHERE sender_id=? AND reward IS NOT NULL",
            (sender_id,),
        ).fetchall()

    if not rows:
        return default

    weighted_sum = 0.0
    weight_total = 0.0
    for reward, ts in rows:
        age = max(0.0, now - ts)
        weight = math.pow(0.5, age / DECAY_HALF_LIFE_S)
        weighted_sum += reward * weight
        weight_total += weight

    if weight_total == 0:
        return default
    return max(0.0, min(1.0, weighted_sum / weight_total))


def get_sender_stats(sender_id: str) -> dict:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*), AVG(CASE WHEN reward IS NOT NULL THEN reward END) FROM interactions WHERE sender_id=?",
            (sender_id,),
        ).fetchone()
    total, avg_reward = row
    return {
        "sender_id": sender_id,
        "total_interactions": total or 0,
        "avg_reward": round(avg_reward, 3) if avg_reward is not None else None,
        "affinity": round(get_sender_affinity(sender_id), 3),
    }


def list_senders(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT sender_id FROM senders ORDER BY last_seen DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [get_sender_stats(r[0]) for r in rows]


def prune_old(days: int = 30) -> int:
    """FIFO prune per SRS Section 6.2 (30-day retention)."""
    cutoff = time.time() - days * 24 * 3600
    with _connect() as conn:
        cur = conn.execute("DELETE FROM interactions WHERE ts < ?", (cutoff,))
        conn.commit()
        return cur.rowcount


init_db()
