"""Models survive a restart on an ephemeral filesystem, and the schema self-updates."""

from sqlalchemy import inspect, text

import app.services.prediction_service as prediction_service
from app.migrations import run_migrations
from app.ml.model_store import publish_models, restore_model
from app.models import ModelArtifact


def test_publish_stores_only_that_sports_files(db_session, tmp_path) -> None:
    (tmp_path / "match_winner.joblib").write_bytes(b"football-model")
    (tmp_path / "training_report.json").write_text("{}")
    (tmp_path / "basketball_winner.joblib").write_bytes(b"basketball-model")

    assert publish_models(tmp_path, "football") == ["match_winner.joblib", "training_report.json"]
    assert publish_models(tmp_path, "basketball") == ["basketball_winner.joblib"]
    assert db_session.get(ModelArtifact, "match_winner.joblib").data == b"football-model"


def test_republishing_replaces_the_stored_file(db_session, tmp_path) -> None:
    model = tmp_path / "btts.joblib"
    model.write_bytes(b"old")
    publish_models(tmp_path, "football")
    model.write_bytes(b"newer")
    publish_models(tmp_path, "football")

    db_session.expire_all()
    row = db_session.get(ModelArtifact, "btts.joblib")
    assert (row.data, row.size_bytes) == (b"newer", 5)


def test_restore_writes_the_file_back(db_session, tmp_path) -> None:
    (tmp_path / "btts.joblib").write_bytes(b"trained")
    publish_models(tmp_path, "football")

    fresh = tmp_path / "after-restart"
    assert restore_model("btts.joblib", fresh) is True
    assert (fresh / "btts.joblib").read_bytes() == b"trained"
    assert restore_model("never_trained.joblib", fresh) is False


def test_prediction_loads_a_model_that_only_exists_in_the_database(
    db_session, trained_models, tmp_path, monkeypatch
) -> None:
    publish_models(tmp_path, "football")
    restarted = tmp_path / "wiped"
    monkeypatch.setattr(prediction_service, "MODEL_DIR", str(restarted))
    prediction_service.clear_model_cache()

    bundle = prediction_service._load("match_winner")

    assert "model" in bundle
    assert (restarted / "match_winner.joblib").exists()


def test_store_failure_never_breaks_training_or_inference(tmp_path) -> None:
    # No db_session fixture: the tables do not exist, so every query fails.
    (tmp_path / "btts.joblib").write_bytes(b"x")
    assert publish_models(tmp_path, "football") == []
    assert restore_model("btts.joblib", tmp_path / "empty") is False


def test_migrations_add_columns_missing_from_an_older_database(db_session) -> None:
    engine = db_session.get_bind()
    with engine.begin() as connection:
        connection.execute(text("DROP INDEX ix_matches_sport"))
        connection.execute(text("ALTER TABLE betting_slips DROP COLUMN odds_are_estimates"))

    applied = run_migrations(engine)

    assert "betting_slips.odds_are_estimates: added" in applied
    assert "ix_matches_sport: created" in applied
    columns = {c["name"] for c in inspect(engine).get_columns("betting_slips")}
    assert "odds_are_estimates" in columns
    assert run_migrations(engine) == []
