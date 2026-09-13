"""End-to-end pipeline test on a synthetic league with known team strengths.

Real football data is ~55% predictable at best, so asserting on absolute
accuracy would be meaningless. Instead this simulates a league where a genuine
signal exists, then asserts the pipeline recovers it - and, crucially, that the
temporal split and calibration behave.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ml.backtest import calibration_table, value_simulation, walk_forward
from app.ml.features import FEATURE_COLUMNS, build_dataset
from app.ml.train import temporal_split, train_models

N_TEAMS = 20
N_MATCHES = 1500


def _simulate_league(seed: int = 7) -> list[dict]:
    """Poisson scorelines driven by latent per-team attack/defence strengths."""
    rng = np.random.default_rng(seed)
    attack = rng.normal(1.35, 0.35, N_TEAMS).clip(0.5, 2.6)
    defence = rng.normal(1.0, 0.22, N_TEAMS).clip(0.5, 1.8)

    matches = []
    start = datetime(2021, 8, 1)
    for i in range(N_MATCHES):
        home, away = rng.choice(N_TEAMS, size=2, replace=False)
        home_rate = attack[home] * defence[away] * 1.25  # home advantage
        away_rate = attack[away] * defence[home]
        hg, ag = int(rng.poisson(home_rate)), int(rng.poisson(away_rate))
        matches.append({
            "id": i,
            "league_id": 1,
            "season": 2021 + i // 500,
            "home_team_id": int(home),
            "away_team_id": int(away),
            "kickoff_time": start + timedelta(hours=8 * i),
            "home_goals": hg,
            "away_goals": ag,
            "home_shots_on_target": int(rng.poisson(home_rate * 3)),
            "away_shots_on_target": int(rng.poisson(away_rate * 3)),
            "home_possession": float(rng.normal(52, 6)),
            "away_possession": float(rng.normal(48, 6)),
            # Odds roughly track true strength, with a bookmaker margin.
            "odds_home": float(1 / max(0.05, min(0.9, home_rate / (home_rate + away_rate + 0.8))) * 0.93),
            "odds_draw": 3.6,
            "odds_away": float(1 / max(0.05, min(0.9, away_rate / (home_rate + away_rate + 0.8))) * 0.93),
        })
    return matches


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return build_dataset(_simulate_league())


def test_dataset_has_expected_shape(dataset: pd.DataFrame) -> None:
    assert len(dataset) == N_MATCHES
    assert not dataset[FEATURE_COLUMNS].isna().any().any(), "features must never contain NaN"
    assert set(dataset["match_winner"].unique()) == {"H", "D", "A"}


def test_temporal_split_keeps_test_set_in_the_future(dataset: pd.DataFrame) -> None:
    train, test = temporal_split(dataset)
    assert train["kickoff_time"].max() <= test["kickoff_time"].min()
    assert len(test) == pytest.approx(len(dataset) * 0.2, rel=0.05)


def test_training_beats_the_base_rate_baseline(dataset: pd.DataFrame, tmp_path, monkeypatch) -> None:
    import app.ml.train as train_module

    monkeypatch.setattr(train_module, "MODEL_DIR", str(tmp_path))
    summary = train_models(dataset, targets=["match_winner"])
    report = summary["targets"][0]

    assert report["beats_baseline"], (
        f"log_loss {report['log_loss']} did not beat baseline {report['baseline_log_loss']}"
    )
    assert (tmp_path / "match_winner.joblib").exists()
    assert (tmp_path / "training_report.json").exists()


def test_trained_model_serves_predictions(dataset: pd.DataFrame, tmp_path, monkeypatch) -> None:
    import app.ml.train as train_module
    from app.services import prediction_service

    monkeypatch.setattr(train_module, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(prediction_service, "MODEL_DIR", str(tmp_path))
    prediction_service.clear_model_cache()
    train_models(dataset, targets=["match_winner"])

    features = dataset[FEATURE_COLUMNS].iloc[0].to_dict()
    result = prediction_service.infer_match_winner(features)

    assert result["prediction"] in {"H", "D", "A"}
    assert 0 < result["confidence"] <= 100
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    prediction_service.clear_model_cache()


def test_training_refuses_a_dataset_that_is_too_small() -> None:
    tiny = build_dataset(_simulate_league()[:50])
    with pytest.raises(ValueError, match="at least"):
        train_models(tiny)


def test_walk_forward_backtest_reports_every_fold(dataset: pd.DataFrame) -> None:
    result = walk_forward(dataset, folds=3, min_train=600)
    assert len(result["folds"]) == 3
    assert 0 < result["mean_accuracy"] < 1
    assert not result["predictions"].empty


def test_value_simulation_reports_roi(dataset: pd.DataFrame) -> None:
    result = walk_forward(dataset, folds=3, min_train=600)
    simulation = value_simulation(result["predictions"], edge_threshold=0.03)
    # Either it found selections and reported ROI, or it honestly found none.
    assert "roi_percent" in simulation or simulation["bets"] == 0


def test_calibration_table_is_monotonic_enough(dataset: pd.DataFrame) -> None:
    result = walk_forward(dataset, folds=3, min_train=600)
    table = calibration_table(result["predictions"])
    assert not table.empty
    # Buckets with a decent sample should track the diagonal within 15 points.
    sizeable = table[table["n"] >= 30]
    assert (abs(sizeable["predicted"] - sizeable["observed"]) < 0.15).all()
