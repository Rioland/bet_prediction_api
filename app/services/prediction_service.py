"""Model inference.

Returns calibrated probabilities. ``confidence`` is the probability of the most
likely outcome - it is a real probability, not a marketing score, and should be
presented that way.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException

from app.core.config import settings


@lru_cache(maxsize=8)
def _load(target: str) -> dict[str, Any]:
    model_path = Path(settings.model_dir) / f"{target}.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model '{target}' is not trained yet. Run scripts/train_model.py.",
        )
    return joblib.load(model_path)


def clear_model_cache() -> None:
    _load.cache_clear()


def predict(target: str, match_features: dict[str, float]) -> dict[str, Any]:
    bundle = _load(target)
    model, features, classes = bundle["model"], bundle["features"], bundle["classes"]

    missing = [f for f in features if f not in match_features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing[:5]}")

    # A DataFrame keeps the column names the pipeline was fitted with, so a
    # reordered feature dict can never silently shift values between columns.
    vector = pd.DataFrame([{f: float(match_features[f]) for f in features}], columns=features)
    probabilities = model.predict_proba(vector)[0]
    ranked = sorted(zip(classes, probabilities), key=lambda pair: pair[1], reverse=True)
    top_label, top_probability = ranked[0]

    return {
        "prediction": str(top_label),
        "confidence": round(float(top_probability) * 100, 2),
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, probabilities)},
    }


def infer_match_winner(match_features: dict[str, float]) -> dict[str, Any]:
    return predict("match_winner", match_features)
