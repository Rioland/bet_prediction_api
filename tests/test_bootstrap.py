"""A new deployment fills itself; a running one is left alone."""

import asyncio

import pytest

import app.services.bootstrap as bootstrap_module
from app.ml.train import MIN_ROWS
from app.services.bootstrap import bootstrap


class Server:
    """Stands in for the database, the feed and the trainer."""

    def __init__(self, completed: int, model: bool, backfill_adds: int = 0) -> None:
        self.completed = completed
        self.model = model
        self.backfill_adds = backfill_adds
        self.backfilled_days: list[int] = []
        self.trained: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "Server":
        async def backfill(days: int) -> int:
            self.backfilled_days.append(days)
            self.completed += self.backfill_adds
            return self.backfill_adds

        def retrain(sport: str) -> None:
            self.trained.append(sport)
            self.model = True

        monkeypatch.setattr(bootstrap_module, "_completed_count", lambda sport: self.completed)
        monkeypatch.setattr(bootstrap_module, "_model_stored", lambda sport: self.model)
        monkeypatch.setattr(bootstrap_module, "_backfill", backfill)
        monkeypatch.setattr(bootstrap_module, "_retrain", retrain)
        return self


def test_empty_database_is_backfilled_then_trained(monkeypatch: pytest.MonkeyPatch) -> None:
    server = Server(completed=0, model=False, backfill_adds=MIN_ROWS + 300).install(monkeypatch)

    asyncio.run(bootstrap())

    assert server.backfilled_days == [bootstrap_module.BOOTSTRAP_DAYS]
    assert server.trained == ["football"]


def test_populated_deployment_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    server = Server(completed=MIN_ROWS + 1000, model=True).install(monkeypatch)

    asyncio.run(bootstrap())

    assert server.backfilled_days == []
    assert server.trained == []


def test_history_without_a_model_trains_without_backfilling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = Server(completed=MIN_ROWS + 1000, model=False).install(monkeypatch)

    asyncio.run(bootstrap())

    assert server.backfilled_days == []
    assert server.trained == ["football"]


def test_too_little_history_does_not_train(monkeypatch: pytest.MonkeyPatch) -> None:
    # The feed came back thin; training on it would publish a model that has
    # learned nothing, so it waits for more results instead.
    server = Server(completed=0, model=False, backfill_adds=MIN_ROWS - 1).install(monkeypatch)

    asyncio.run(bootstrap())

    assert server.backfilled_days == [bootstrap_module.BOOTSTRAP_DAYS]
    assert server.trained == []


def test_a_failure_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    Server(completed=0, model=False).install(monkeypatch)

    async def feed_down(days: int) -> int:
        raise ConnectionError("ESPN unreachable")

    monkeypatch.setattr(bootstrap_module, "_backfill", feed_down)

    asyncio.run(bootstrap())  # must not raise: it runs beside the live API

    assert "Startup bootstrap for football failed" in caplog.text
