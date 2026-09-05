"""Dixon-Coles scoreline model.

The value of this model is internal consistency, so most of these tests assert
relationships between markets rather than absolute numbers.
"""

import pytest

from app.ml.dixon_coles import (
    MAX_GOALS,
    expected_goals,
    outcomes,
    over_line,
    predict,
    score_matrix,
)

FEATURES = {
    "home_goals_scored_avg": 1.7,
    "away_goals_scored_avg": 1.1,
    "home_goals_conceded_avg": 1.0,
    "away_goals_conceded_avg": 1.5,
}


def test_score_matrix_is_a_probability_distribution() -> None:
    matrix = score_matrix(1.6, 1.1)
    total = sum(sum(row) for row in matrix)
    assert total == pytest.approx(1.0, abs=1e-9)
    assert all(cell >= 0 for row in matrix for cell in row)


def test_outcomes_partition_the_distribution() -> None:
    """Home win, draw and away win are exhaustive and mutually exclusive."""
    result = outcomes(score_matrix(1.6, 1.1))
    assert result["home_win"] + result["draw"] + result["away_win"] == pytest.approx(1.0, abs=1e-9)


def test_goal_lines_are_monotonic_across_the_same_matrix() -> None:
    matrix = score_matrix(1.5, 1.2)
    lines = [over_line(matrix, line) for line in (0.5, 1.5, 2.5, 3.5, 4.5)]
    assert lines == sorted(lines, reverse=True), "a higher line must be less likely"


def test_stronger_home_side_raises_its_win_probability() -> None:
    weak = outcomes(score_matrix(1.0, 1.4))
    strong = outcomes(score_matrix(2.4, 1.0))
    assert strong["home_win"] > weak["home_win"]
    assert strong["away_win"] < weak["away_win"]


def test_low_score_correction_shifts_mass_versus_plain_poisson() -> None:
    """The tau correction is the whole point - it must actually do something."""
    from math import exp

    lam_h, lam_a = 1.3, 1.1
    matrix = score_matrix(lam_h, lam_a)

    def plain(k: float, lam: float) -> float:
        from math import factorial
        return (lam**k) * exp(-lam) / factorial(k)

    independent_draw_11 = plain(1, lam_h) * plain(1, lam_a)
    assert matrix[1][1] != pytest.approx(independent_draw_11, abs=1e-4)


def test_draw_rate_is_football_plausible() -> None:
    # Real-world 1X2 draw rates sit around 25%.
    result = outcomes(score_matrix(1.5, 1.2))
    assert 0.20 < result["draw"] < 0.32


def test_evenly_matched_sides_produce_a_symmetric_market() -> None:
    result = outcomes(score_matrix(1.3, 1.3))
    assert result["home_win"] == pytest.approx(result["away_win"], abs=1e-9)


def test_extreme_rates_are_clamped_not_exploded() -> None:
    matrix = score_matrix(50.0, -3.0)
    assert sum(sum(row) for row in matrix) == pytest.approx(1.0, abs=1e-9)


def test_expected_goals_favour_the_stronger_attack() -> None:
    home_xg, away_xg = expected_goals(FEATURES)
    assert home_xg > away_xg
    assert 0 < away_xg < 5 and 0 < home_xg < 5


def test_predict_returns_every_market_consistently() -> None:
    result = predict(FEATURES)
    assert result["home_win"] + result["draw"] + result["away_win"] == pytest.approx(1.0, abs=1e-9)
    assert result["over_1_5"] >= result["over_2_5"] >= result["over_3_5"]
    assert "-" in result["predicted_score"]


def test_matrix_covers_the_realistic_scoreline_range() -> None:
    assert MAX_GOALS >= 9, "truncating below 9 goals loses non-trivial mass"
