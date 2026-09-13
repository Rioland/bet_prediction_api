"""Model inference.

Returns calibrated probabilities. ``confidence`` is the probability of the most
likely outcome - a real probability, not a marketing score, and should be
presented that way.

Isotonic calibration assigns a hard zero to any class it never saw win in a
calibration bin, which then propagates: a 1X2 split of 0.78/0.22/0.00 makes
double chance exactly 1.0, and the UI reports a 100% certain bet. No football
outcome is impossible, so probabilities are floored and renormalised.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException

from app.config import MODEL_DIR

# No outcome is impossible. Half a percentage point is below anything the site
# would surface as a tip, but keeps derived markets off 0 and 100.
MIN_CLASS_PROBABILITY = 0.005


def floor_probabilities(probabilities: list[float]) -> list[float]:
    """Lift near-zero classes to the floor, taking the difference from the rest.

    Scaling every class after flooring would push the floored ones back under
    the floor. Instead the remaining mass is redistributed across the classes
    that were already above it, so the floor genuinely holds and the total is
    still one.
    """
    values = [float(p) for p in probabilities]
    below = [i for i, p in enumerate(values) if p < MIN_CLASS_PROBABILITY]
    if not below:
        return values

    above = [i for i in range(len(values)) if i not in below]
    if not above:  # every class under the floor; fall back to uniform
        return [1 / len(values)] * len(values)

    reserved = len(below) * MIN_CLASS_PROBABILITY
    remaining = sum(values[i] for i in above)
    scale = (1.0 - reserved) / remaining if remaining > 0 else 0.0

    result = list(values)
    for i in below:
        result[i] = MIN_CLASS_PROBABILITY
    for i in above:
        result[i] = values[i] * scale
    return result


@lru_cache(maxsize=16)
def _load(target: str, sport: str = "football") -> dict[str, Any]:
    from app.ml.train import model_filename

    model_path = Path(MODEL_DIR) / model_filename(sport, target)
    if not model_path.exists():
        # An ephemeral filesystem loses ./models on restart; the database keeps
        # a copy of the last trained models.
        from app.ml.model_store import restore_model

        restore_model(model_path.name, MODEL_DIR)
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained '{target}' model for {sport}. "
                f"Run: python scripts/train_model.py --sport {sport}"
            ),
        )
    return joblib.load(model_path)


def clear_model_cache() -> None:
    _load.cache_clear()


def predict(
    target: str, match_features: dict[str, float], sport: str = "football"
) -> dict[str, Any]:
    bundle = _load(target, sport)
    model, features, classes = bundle["model"], bundle["features"], bundle["classes"]

    missing = [f for f in features if f not in match_features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing[:5]}")

    # A DataFrame keeps the column names the pipeline was fitted with, so a
    # reordered feature dict can never silently shift values between columns.
    vector = pd.DataFrame([{f: float(match_features[f]) for f in features}], columns=features)
    probabilities = floor_probabilities(model.predict_proba(vector)[0])
    ranked = sorted(zip(classes, probabilities), key=lambda pair: pair[1], reverse=True)
    top_label, top_probability = ranked[0]

    return {
        "prediction": str(top_label),
        "confidence": round(float(top_probability) * 100, 2),
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, probabilities)},
    }


def predict_many(
    target: str, feature_rows: list[dict[str, float]], sport: str = "football"
) -> list[dict[str, Any]]:
    """Predict a batch in one pass.

    A calibrated forest costs ~100ms per call regardless of how many rows it is
    given, almost all of it fixed overhead. Scoring a fifty-fixture card one row
    at a time took ten seconds; the same work batched takes well under one.
    """
    if not feature_rows:
        return []

    bundle = _load(target, sport)
    model, features, classes = bundle["model"], bundle["features"], bundle["classes"]

    missing = [f for f in features if f not in feature_rows[0]]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing[:5]}")

    frame = pd.DataFrame(
        [{f: float(row[f]) for f in features} for row in feature_rows], columns=features
    )
    matrix = model.predict_proba(frame)

    results = []
    for probabilities in matrix:
        floored = floor_probabilities(probabilities)
        top_label, top_probability = max(zip(classes, floored), key=lambda pair: pair[1])
        results.append({
            "prediction": str(top_label),
            "confidence": round(float(top_probability) * 100, 2),
            "probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, floored)},
        })
    return results


def infer_match_winner(match_features: dict[str, float]) -> dict[str, Any]:
    return predict("match_winner", match_features)
