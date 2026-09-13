"""Basketball adapter.

Markets are deliberately narrower than football's. Without a scoreline
distribution there is nothing principled behind correct-score or
both-teams-to-score equivalents, so only the markets the model actually
predicts are offered.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.ml import features_basketball


class BasketballAdapter:
    name = "basketball"
    display_name = "Basketball"
    has_draw = False
    feature_columns = features_basketball.FEATURE_COLUMNS
    targets = features_basketball.TARGETS
    # Schedules are dense, so history accrues fast and a slightly higher bar
    # is affordable.
    min_team_history = 8

    def build_dataset(self, matches: list[dict[str, Any]]) -> pd.DataFrame:
        return features_basketball.build_dataset(matches)

    def features_for_upcoming(
        self, finished: list[dict[str, Any]], upcoming: list[dict[str, Any]]
    ) -> pd.DataFrame:
        return features_basketball.features_for_upcoming(finished, upcoming)

    def settle(self, market: str, selection: str, home_score: int, away_score: int) -> str:
        total = home_score + away_score
        won = {
            ("moneyline", "1"): home_score > away_score,
            ("moneyline", "2"): away_score > home_score,
            ("total_points", "over"): total > features_basketball.DEFAULT_TOTAL_LINE,
            ("total_points", "under"): total < features_basketball.DEFAULT_TOTAL_LINE,
        }.get((market, selection))

        if won is None:
            return "void"
        return "won" if won else "lost"
