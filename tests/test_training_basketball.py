"""End-to-end training for a second sport.

Asserts the pipeline is genuinely sport-agnostic: the same trainer, fed a
different adapter, produces a working model without touching football's.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ml.train import model_filename, report_filename, train_for_sport
from app.sports.registry import get_adapter

N_TEAMS = 16
N_GAMES = 1400


def _simulate_season(seed: int = 5) -> list[dict]:
    """Games driven by latent team strength, with a home edge and real noise."""
    rng = np.random.default_rng(seed)
    offence = rng.normal(112, 6, N_TEAMS)
    defence = rng.normal(112, 5, N_TEAMS)

    games = []
    start = datetime(2026, 10, 1, 19, 0)
    for i in range(N_GAMES):
        home, away = rng.choice(N_TEAMS, size=2, replace=False)
        home_pts = int(rng.normal((offence[home] + defence[away]) / 2 + 2.8, 11))
        away_pts = int(rng.normal((offence[away] + defence[home]) / 2, 11))
        if home_pts == away_pts:
            home_pts += 1  # basketball has no draw
        games.append({
            "id": i,
            "league_id": 1,
            "home_team_id": int(home),
            "away_team_id": int(away),
            "kickoff_time": start + timedelta(hours=12 * i),
            "home_goals": home_pts,
            "away_goals": away_pts,
            "odds_home": 1.85,
            "odds_away": 1.95,
        })
    return games


@pytest.fixture(scope="module")
def basketball_frame() -> pd.DataFrame:
    return get_adapter("basketball").build_dataset(_simulate_season())


def test_dataset_has_no_draws_and_no_nans(basketball_frame) -> None:
    adapter = get_adapter("basketball")
    assert set(basketball_frame["match_winner"].unique()) == {"H", "A"}
    assert not basketball_frame[adapter.feature_columns].isna().any().any()


def test_training_beats_the_base_rate(basketball_frame, tmp_path, monkeypatch) -> None:
    import app.ml.train as train_module

    monkeypatch.setattr(train_module, "MODEL_DIR", str(tmp_path))
    summary = train_module.train_models(
        basketball_frame,
        targets=["match_winner"],
        feature_columns=get_adapter("basketball").feature_columns,
        sport="basketball",
    )
    report = summary["targets"][0]

    assert summary["sport"] == "basketball"
    assert report["beats_baseline"], (
        f"log_loss {report['log_loss']} did not beat baseline {report['baseline_log_loss']}"
    )


def test_artifacts_are_namespaced_so_sports_cannot_collide(
    basketball_frame, tmp_path, monkeypatch
) -> None:
    import app.ml.train as train_module

    monkeypatch.setattr(train_module, "MODEL_DIR", str(tmp_path))
    train_for_sport(get_adapter("basketball"), basketball_frame)

    assert (tmp_path / model_filename("basketball", "match_winner")).exists()
    assert (tmp_path / report_filename("basketball")).exists()
    # Football's unprefixed artifacts must not have been written or clobbered.
    assert not (tmp_path / model_filename("football", "match_winner")).exists()
    assert not (tmp_path / report_filename("football")).exists()


def test_saved_bundle_records_its_sport_and_features(
    basketball_frame, tmp_path, monkeypatch
) -> None:
    """Inference must be able to reject a model fed the wrong sport's rows."""
    import joblib

    import app.ml.train as train_module

    monkeypatch.setattr(train_module, "MODEL_DIR", str(tmp_path))
    train_module.train_models(
        basketball_frame,
        targets=["match_winner"],
        feature_columns=get_adapter("basketball").feature_columns,
        sport="basketball",
    )

    bundle = joblib.load(tmp_path / model_filename("basketball", "match_winner"))
    assert bundle["sport"] == "basketball"
    assert bundle["features"] == get_adapter("basketball").feature_columns
    assert set(bundle["classes"]) == {"H", "A"}
