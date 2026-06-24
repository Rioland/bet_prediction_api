from pathlib import Path

import joblib
import numpy as np
from fastapi import HTTPException

from app.core.config import settings
from app.ml.train import FEATURE_COLUMNS


def infer_match_winner(match_features: dict) -> dict:
    model_path = Path(settings.model_dir) / "match_winner.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=503, detail="Prediction model not trained")

    model = joblib.load(model_path)
    vector = np.array([[match_features.get(k, 0) for k in FEATURE_COLUMNS]])
    pred = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]
    class_map = {str(c): float(p) for c, p in zip(model.classes_, probs, strict=True)}
    confidence = max(class_map.values()) * 100
    return {"prediction": str(pred), "confidence": round(confidence, 2), "probabilities": class_map}
