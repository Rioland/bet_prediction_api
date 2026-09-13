"""Per-sport behaviour behind one interface.

Football and basketball differ in ways that reach the model, not just the
labels: football has draws and low scores where a Poisson model is reasonable,
basketball has no draw and scores in the 90s where it is not. Rather than
branch on sport inside training and inference, each sport supplies its own
feature builder, targets and settlement rules, and the rest of the pipeline
stays sport-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class SportAdapter(Protocol):
    """Everything the pipeline needs to know about one sport."""

    #: Identifier used in routes, the database and model filenames.
    name: str

    #: Human-readable, for API responses.
    display_name: str

    #: Whether a drawn result is possible. Drives the number of outcome classes.
    has_draw: bool

    #: Feature column names, in the order the model expects them.
    feature_columns: list[str]

    #: Trainable targets, e.g. ["match_winner", "total_points_over"].
    targets: list[str]

    #: Minimum completed matches per team before a fixture can be predicted.
    min_team_history: int

    def build_dataset(self, matches: list[dict[str, Any]]) -> pd.DataFrame:
        """Labelled training rows from completed matches, in time order.

        Must be leak-free: a row's features may only reflect matches that
        started before it.
        """
        ...

    def features_for_upcoming(
        self, finished: list[dict[str, Any]], upcoming: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Feature rows for fixtures that have not been played."""
        ...

    def settle(self, market: str, selection: str, home_score: int, away_score: int) -> str:
        """Score a published selection: "won" | "lost" | "void"."""
        ...
