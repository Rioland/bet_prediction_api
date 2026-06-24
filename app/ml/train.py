from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.core.config import settings

FEATURE_COLUMNS = [
    "home_form",
    "away_form",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "home_goals_scored_avg",
    "away_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    "home_league_position",
    "away_league_position",
    "home_xg",
    "away_xg",
    "home_possession",
    "away_possession",
    "home_shots_on_target",
    "away_shots_on_target",
]


def train_models(csv_path: str) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    for target in ["match_winner", "over_under_2_5", "btts", "correct_score"]:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        models = {
            "rf": RandomForestClassifier(n_estimators=250, random_state=42),
            "xgb": XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
            ),
            "lr": LogisticRegression(max_iter=1000),
        }

        best_name = "rf"
        best_score = -1.0
        best_model = None
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            if score > best_score:
                best_name = name
                best_score = score
                best_model = model
        assert best_model is not None
        print(f"[{target}] using={best_name} score={best_score:.3f}")
        print(classification_report(y_test, best_model.predict(X_test)))
        out_path = model_dir / f"{target}.joblib"
        joblib.dump(best_model, out_path)
        artifacts[target] = str(out_path)
    return artifacts
