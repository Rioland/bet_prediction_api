"""Football adapter over the existing pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.ml import features as football_features
from app.services.results import settle_selection


class FootballAdapter:
    name = "football"
    display_name = "Football"
    has_draw = True
    feature_columns = football_features.FEATURE_COLUMNS
    targets = football_features.TARGETS
    min_team_history = 5

    def build_dataset(self, matches: list[dict[str, Any]]) -> pd.DataFrame:
        return football_features.build_dataset(matches)

    def features_for_upcoming(
        self, finished: list[dict[str, Any]], upcoming: list[dict[str, Any]]
    ) -> pd.DataFrame:
        return football_features.features_for_upcoming(finished, upcoming)

    def settle(self, market: str, selection: str, home_score: int, away_score: int) -> str:
        return settle_selection(market, selection, home_score, away_score)
