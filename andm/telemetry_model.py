"""
telemetry_model.py
-------------------
Implements SRS Section 4.2: "Telemetry Synchronization and Context Feature
Fusion" using a LightGBM Gradient Boosted Decision Tree regressor
(SRS 3.3.1 names LightGBM C-API as the required engine; we use the
LightGBM Python API here, which is the correct choice for the web-based
simulation layer described in SRS Section 1.5).

INPUT (the telemetry feature vector, SRS 3.2.1 / REQ-TEL-1..3):
  speed_mps            - GPS speed, polled per REQ-TEL-2 throttling rule
  imu_magnitude        - accelerometer magnitude from the 20 Hz IMU cache
                          (REQ-TEL-1), proxy for how physically active/
                          in-motion the user currently is
  is_charging           - power state flag
  is_unmetered_wifi     - connectivity flag
  calendar_busy          - user currently has a calendar event in progress
  hour_of_day            - local hour, 0-23

OUTPUT:
  context_score in [0, 1]: how *receptive* the user is to an interruption
  right now. High = idle/available (deliver freely). Low = driving fast,
  in a meeting, or asleep (hold / digest).

TRAINING DATA NOTE (honest, on purpose):
No real on-device telemetry logs exist yet for this project (data
collection requires a shipped mobile build, which is explicitly
out-of-scope until the prototype validates the simulator -- SRS 1.3
Justification Strategy #2/#4). We bootstrap the GBDT with a synthetic
dataset generated directly from the business rules already fixed in the
SRS (Section 5.5 "Business Rules", Section 2.5 constraints): fast
movement, an in-progress calendar event, or very early/late hours lower
receptivity; idle + charging + unmetered wifi (the same conditions that
gate the federated learning loop, SRS 2.5) raises it. This gives the
routing pipeline a working, swappable component; the TBD list in the SRS
already flags that real empirical telemetry data is needed to finalize
these thresholds (TBD-03 is the closest analogue for privacy epsilon --
an equivalent TBD applies here for the telemetry model weights).
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_LIGHTGBM = False

MODEL_FILE = Path(__file__).parent / "telemetry_gbdt.txt"

FEATURE_NAMES = [
    "speed_mps",
    "imu_magnitude",
    "is_charging",
    "is_unmetered_wifi",
    "calendar_busy",
    "hour_of_day",
]


def _synthetic_dataset(n: int = 6000, seed: int = 42):
    rng = np.random.default_rng(seed)
    speed = rng.exponential(scale=6.0, size=n).clip(0, 40)  # m/s, most near 0
    imu = rng.exponential(scale=1.5, size=n).clip(0, 15)  # m/s^2 magnitude
    charging = rng.integers(0, 2, size=n).astype(float)
    wifi = rng.integers(0, 2, size=n).astype(float)
    calendar_busy = rng.integers(0, 2, size=n).astype(float)
    hour = rng.integers(0, 24, size=n).astype(float)

    # Ground-truth receptivity built from the SRS's own stated business
    # rules + common-sense priors, plus noise.
    score = np.ones(n)
    score -= np.clip(speed / 20.0, 0, 1) * 0.55         # fast movement -> less receptive
    score -= np.clip(imu / 10.0, 0, 1) * 0.20            # high physical activity -> less receptive
    score -= calendar_busy * 0.45                        # in a meeting -> less receptive
    night = ((hour >= 23) | (hour <= 6)).astype(float)
    score -= night * 0.35                                # asleep hours -> less receptive
    score += charging * wifi * 0.05                      # idle-charging-wifi (federated window) -> slightly more receptive
    score += rng.normal(0, 0.05, size=n)                 # noise
    score = np.clip(score, 0.0, 1.0)

    X = np.column_stack([speed, imu, charging, wifi, calendar_busy, hour])
    return X, score


class TelemetryContextModel:
    def __init__(self):
        self.model = None
        self._train()

    def _train(self):
        X, y = _synthetic_dataset()
        if _HAS_LIGHTGBM:
            train_set = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
            params = {
                "objective": "regression",
                "metric": "l2",
                "num_leaves": 15,
                "learning_rate": 0.08,
                "verbose": -1,
                "min_data_in_leaf": 20,
            }
            self.model = lgb.train(params, train_set, num_boost_round=120)
        else:  # pragma: no cover
            self.model = GradientBoostingRegressor(n_estimators=150, max_depth=3)
            self.model.fit(X, y)

    def score(self, speed_mps: float, imu_magnitude: float, is_charging: bool,
              is_unmetered_wifi: bool, calendar_busy: bool, hour_of_day: int) -> float:
        X = np.array([[speed_mps, imu_magnitude, float(is_charging),
                        float(is_unmetered_wifi), float(calendar_busy), float(hour_of_day)]])
        if _HAS_LIGHTGBM:
            pred = self.model.predict(X)[0]
        else:  # pragma: no cover
            pred = self.model.predict(X)[0]
        return float(np.clip(pred, 0.0, 1.0))


# Module-level singleton, trained once at import time (mirrors centroids.pt
# being precomputed once by build.py rather than per-request).
_context_model = TelemetryContextModel()


def get_context_score(speed_mps: float, imu_magnitude: float, is_charging: bool,
                       is_unmetered_wifi: bool, calendar_busy: bool, hour_of_day: int) -> float:
    return _context_model.score(speed_mps, imu_magnitude, is_charging,
                                 is_unmetered_wifi, calendar_busy, hour_of_day)
