"""Model training.

Two things here differ from a textbook sklearn script, and both matter:

1. The train/test split is *temporal*, never random. Football data is a time
   series; a random split lets the model learn from matches that happen after
   the ones it is scored on, which inflates measured accuracy and does not
   survive contact with real fixtures.
2. Models are selected on log loss, not accuracy, and are calibrated. The
   product needs trustworthy probabilities ("this is a 62% pick"), and an
   uncalibrated classifier that is 55% accurate can still be badly wrong about
   its own confidence.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import MODEL_DIR
from app.ml.features import FEATURE_COLUMNS, TARGETS

logger = logging.getLogger(__name__)

# XGBoost needs an OpenMP runtime (libgomp / libomp). The Docker image installs
# it; a local machine may not have it, and a missing booster should degrade the
# candidate list rather than break training entirely. The training report names
# which model actually won, so its absence is visible rather than silent.
try:
    import xgboost  # noqa: F401

    from app.ml.xgb import StringLabelXGB

    XGBOOST_AVAILABLE = True
    _XGBOOST_ERROR = ""
except Exception as exc:  # ImportError, or XGBoostError when libomp is missing
    StringLabelXGB = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False
    _XGBOOST_ERROR = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__

# Fraction of the timeline held out, most recent matches last.
TEST_FRACTION = 0.2
MIN_ROWS = 500


def temporal_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Split chronologically: the model is always tested on the *later* matches."""
    ordered = df.sort_values("kickoff_time").reset_index(drop=True)
    cutoff = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:cutoff], ordered.iloc[cutoff:]


def _candidates() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "logistic": Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=0.5))]
        ),
        "forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1
        ),
        "gradient_boost": HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=42
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["xgboost"] = StringLabelXGB()
    else:
        logger.warning(
            "XGBoost unavailable (%s); training on the remaining candidates.", _XGBOOST_ERROR
        )
    return candidates


def _baseline_log_loss(y_train: pd.Series, y_test: pd.Series, labels: list[str]) -> float:
    """Log loss of always predicting the training-set base rates.

    A model that cannot beat this has learned nothing.
    """
    rates = y_train.value_counts(normalize=True)
    prior = np.array([[rates.get(label, 1e-9) for label in labels]] * len(y_test))
    return float(log_loss(y_test, prior, labels=labels))


def _multiclass_brier(y_true: pd.Series, proba: np.ndarray, labels: list[str]) -> float:
    if len(labels) == 2:
        positive = labels[1]
        return float(brier_score_loss((y_true == positive).astype(int), proba[:, 1]))
    onehot = np.zeros_like(proba)
    for i, label in enumerate(y_true):
        onehot[i, labels.index(label)] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def train_target(
    df: pd.DataFrame,
    target: str,
    model_dir: Path,
    feature_columns: list[str] | None = None,
    sport: str = "football",
) -> dict[str, Any]:
    features = feature_columns or FEATURE_COLUMNS
    train_df, test_df = temporal_split(df)
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    labels = sorted(df[target].unique())

    best: dict[str, Any] | None = None
    for name, estimator in _candidates().items():
        # Calibrate on held-out folds *within* the training window only.
        calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        proba = calibrated.predict_proba(X_test)
        score = float(log_loss(y_test, proba, labels=list(calibrated.classes_)))
        if best is None or score < best["log_loss"]:
            best = {
                "name": name,
                "model": calibrated,
                "log_loss": score,
                "accuracy": float(accuracy_score(y_test, calibrated.predict(X_test))),
                "brier": _multiclass_brier(y_test, proba, list(calibrated.classes_)),
            }

    assert best is not None
    baseline = _baseline_log_loss(y_train, y_test, labels)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best["model"],
            "features": features,
            "classes": list(best["model"].classes_),
            "sport": sport,
        },
        model_dir / model_filename(sport, target),
    )

    return {
        "target": target,
        "sport": sport,
        "model": best["name"],
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "log_loss": round(best["log_loss"], 4),
        "baseline_log_loss": round(baseline, 4),
        "beats_baseline": bool(best["log_loss"] < baseline),
        "accuracy": round(best["accuracy"], 4),
        "brier": round(best["brier"], 4),
    }


def model_filename(sport: str, target: str) -> str:
    """Artifacts are namespaced by sport so two sports cannot overwrite each other.

    Football keeps its original unprefixed names so existing trained models and
    deployments keep working.
    """
    return f"{target}.joblib" if sport == "football" else f"{sport}_{target}.joblib"


def report_filename(sport: str) -> str:
    return "training_report.json" if sport == "football" else f"{sport}_training_report.json"


def train_models(
    df: pd.DataFrame,
    targets: list[str] | None = None,
    feature_columns: list[str] | None = None,
    sport: str = "football",
) -> dict[str, Any]:
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Only {len(df)} labelled matches available; need at least {MIN_ROWS} "
            "for a temporal split to mean anything. Ingest more history first."
        )
    model_dir = Path(MODEL_DIR)
    reports = [
        train_target(df, target, model_dir, feature_columns=feature_columns, sport=sport)
        for target in (targets or TARGETS)
    ]

    summary = {
        "sport": sport,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        "xgboost_available": XGBOOST_AVAILABLE,
        "targets": reports,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / report_filename(sport)).write_text(json.dumps(summary, indent=2))
    return summary


def train_for_sport(adapter: Any, df: pd.DataFrame) -> dict[str, Any]:
    """Train every target a sport declares, using that sport's feature columns."""
    return train_models(
        df,
        targets=adapter.targets,
        feature_columns=adapter.feature_columns,
        sport=adapter.name,
    )
