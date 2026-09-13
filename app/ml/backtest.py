"""Walk-forward backtesting.

A single temporal split says how the model did on one slice of time. A
walk-forward backtest retrains repeatedly as the season progresses, which is
much closer to how the model will actually be used, and reports whether the
predictions would have made or lost money against real closing odds.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from app.ml.features import FEATURE_COLUMNS

_ODDS_COLUMN = {"H": "odds_home", "D": "odds_draw", "A": "odds_away"}


def _fresh_model():
    return CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=42
        ),
        method="isotonic",
        cv=3,
    )


def walk_forward(
    df: pd.DataFrame, target: str = "match_winner", folds: int = 5, min_train: int = 500
) -> dict[str, Any]:
    """Retrain across expanding time windows and score each future block."""
    ordered = df.sort_values("kickoff_time").reset_index(drop=True)
    if len(ordered) < min_train + folds:
        raise ValueError(f"Need more than {min_train + folds} rows to walk forward; have {len(ordered)}")

    step = (len(ordered) - min_train) // folds
    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    for fold in range(folds):
        train_end = min_train + fold * step
        test_end = train_end + step if fold < folds - 1 else len(ordered)
        train, test = ordered.iloc[:train_end], ordered.iloc[train_end:test_end]
        if test.empty or train[target].nunique() < 2:
            continue

        model = _fresh_model()
        model.fit(train[FEATURE_COLUMNS], train[target])
        proba = model.predict_proba(test[FEATURE_COLUMNS])
        predicted = model.predict(test[FEATURE_COLUMNS])

        results.append({
            "fold": fold + 1,
            "train_rows": len(train),
            "test_rows": len(test),
            "accuracy": round(float(accuracy_score(test[target], predicted)), 4),
            "log_loss": round(float(log_loss(test[target], proba, labels=list(model.classes_))), 4),
        })

        block = test.copy()
        block["predicted"] = predicted
        for i, label in enumerate(model.classes_):
            block[f"p_{label}"] = proba[:, i]
        predictions.append(block)

    combined = pd.concat(predictions) if predictions else pd.DataFrame()
    return {
        "folds": results,
        "mean_accuracy": round(float(np.mean([r["accuracy"] for r in results])), 4) if results else None,
        "mean_log_loss": round(float(np.mean([r["log_loss"] for r in results])), 4) if results else None,
        "predictions": combined,
    }


def value_simulation(
    predictions: pd.DataFrame, target: str = "match_winner", edge_threshold: float = 0.05, stake: float = 1.0
) -> dict[str, Any]:
    """Flat-stake simulation over selections where the model disagrees with the market.

    A bet is placed only when the model's probability exceeds the odds-implied
    probability by ``edge_threshold``. Returns ROI - the number that actually
    decides whether a prediction product is worth anything.
    """
    if predictions.empty:
        return {"bets": 0, "note": "no predictions to simulate"}

    staked = returned = wins = 0.0
    bets = 0
    for _, row in predictions.iterrows():
        for label, odds_col in _ODDS_COLUMN.items():
            odds = row.get(odds_col)
            model_p = row.get(f"p_{label}")
            if not odds or odds <= 1.0 or model_p is None or pd.isna(odds) or pd.isna(model_p):
                continue
            implied = 1.0 / float(odds)
            if float(model_p) - implied < edge_threshold:
                continue
            bets += 1
            staked += stake
            if row[target] == label:
                returned += stake * float(odds)
                wins += 1

    if bets == 0:
        return {"bets": 0, "note": "no selections cleared the edge threshold"}
    return {
        "bets": bets,
        "staked": round(staked, 2),
        "returned": round(returned, 2),
        "profit": round(returned - staked, 2),
        "roi_percent": round((returned - staked) / staked * 100, 2),
        "strike_rate_percent": round(wins / bets * 100, 2),
        "edge_threshold": edge_threshold,
    }


def calibration_table(predictions: pd.DataFrame, label: str = "H", bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs observed frequency - the honesty check.

    If the model says 70% and those matches land 70% of the time, the number
    shown on the site means what it claims to mean.
    """
    column = f"p_{label}"
    if predictions.empty or column not in predictions:
        return pd.DataFrame()
    df = predictions.copy()
    df["bucket"] = pd.cut(df[column], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = df.groupby("bucket", observed=True).agg(
        predicted=(column, "mean"),
        observed=("match_winner", lambda s: float((s == label).mean())),
        n=(column, "size"),
    )
    return grouped.reset_index()
