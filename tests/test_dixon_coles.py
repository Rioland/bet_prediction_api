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


# --- first half and handicap -------------------------------------------------


def test_first_half_has_fewer_goals_than_the_match() -> None:
    from app.ml.dixon_coles import first_half

    half = first_half(1.9, 0.95)
    assert half["expected_goals"] < 1.9 + 0.95


def test_a_level_first_half_is_likelier_than_a_level_match() -> None:
    """Fewer goals in a shorter period means more draws - the sanity check."""
    from app.ml.dixon_coles import first_half

    half = first_half(1.9, 0.95)
    full = outcomes(score_matrix(1.9, 0.95))
    assert half["draw"] > full["draw"]


def test_first_half_outcomes_form_a_distribution() -> None:
    from app.ml.dixon_coles import first_half

    half = first_half(1.6, 1.2)
    assert half["home_win"] + half["draw"] + half["away_win"] == pytest.approx(1.0, abs=1e-9)
    assert half["over_0_5"] >= half["over_1_5"]


def test_minus_half_handicap_equals_the_outright_win() -> None:
    """A -0.5 handicap is the win market by another name; it must agree exactly."""
    from app.ml.dixon_coles import handicap

    assert handicap(1.9, 0.95, -0.5, "home") == pytest.approx(
        outcomes(score_matrix(1.9, 0.95))["home_win"], abs=1e-9
    )


def test_handicap_is_monotonic_in_the_line() -> None:
    from app.ml.dixon_coles import handicap

    lines = [handicap(1.9, 0.95, line, "home") for line in (-2.5, -1.5, -0.5, 0.5, 1.5)]
    assert lines == sorted(lines), "a more generous line cannot be less likely"


def test_giving_start_beats_receiving_it_for_the_favourite() -> None:
    from app.ml.dixon_coles import handicap

    assert handicap(1.9, 0.95, -1.5, "home") < handicap(1.9, 0.95, 1.5, "home")


def test_whole_number_handicaps_are_refused() -> None:
    """A whole line can end level and push, returning the stake - one
    probability cannot express win/lose/push."""
    from app.ml.dixon_coles import handicap

    with pytest.raises(ValueError, match="push"):
        handicap(1.9, 0.95, -1.0, "home")


# --- leading, nil and the shape of the goals ---------------------------------


def test_leading_at_some_point_is_at_least_winning() -> None:
    """Every winner led at the final whistle; some who led did not win."""
    from app.ml.dixon_coles import ever_leads

    full = outcomes(score_matrix(1.9, 0.95))
    assert ever_leads(1.9, 0.95, "home") > full["home_win"]
    assert ever_leads(1.9, 0.95, "away") > full["away_win"]


def test_evenly_matched_sides_are_equally_likely_to_lead() -> None:
    from app.ml.dixon_coles import ever_leads

    assert ever_leads(1.3, 1.3, "home") == pytest.approx(ever_leads(1.3, 1.3, "away"))


def test_leading_rises_with_the_scoring_rate() -> None:
    from app.ml.dixon_coles import ever_leads

    rates = [ever_leads(rate, 1.2, "home") for rate in (0.6, 1.2, 2.4)]
    assert rates == sorted(rates), "a better attack cannot lead less often"


def test_nobody_leads_in_a_goalless_match() -> None:
    """Leading requires a goal, so no side can lead more often than one is scored.

    The two sides are not bounded jointly: a match that goes 1-0 then 1-2 has
    both of them leading, so their probabilities can sum past one.
    """
    from math import exp

    from app.ml.dixon_coles import ever_leads

    any_goal = 1 - exp(-(1.4 + 1.1))
    assert ever_leads(1.4, 1.1, "home") <= any_goal + 1e-9
    assert ever_leads(1.4, 1.1, "away") <= any_goal + 1e-9


def test_winning_to_nil_implies_a_clean_sheet() -> None:
    from app.ml.dixon_coles import clean_sheet, win_to_nil

    matrix = score_matrix(1.9, 0.95)
    assert win_to_nil(matrix, "home") < clean_sheet(matrix, "home")


def test_a_clean_sheet_is_a_win_to_nil_or_a_goalless_draw() -> None:
    from app.ml.dixon_coles import clean_sheet, win_to_nil

    matrix = score_matrix(1.6, 1.1)
    assert clean_sheet(matrix, "home") == pytest.approx(win_to_nil(matrix, "home") + matrix[0][0])


def test_half_time_full_time_is_a_distribution() -> None:
    from app.ml.dixon_coles import half_time_full_time

    combos = half_time_full_time(1.9, 0.95)
    assert len(combos) == 9
    assert sum(combos.values()) == pytest.approx(1.0, abs=1e-9)


def test_coming_from_behind_is_rarer_than_holding_on() -> None:
    """The favourite losing the first half and winning the match is the hard way."""
    from app.ml.dixon_coles import half_time_full_time

    combos = half_time_full_time(1.9, 0.95)
    assert combos["away/home"] < combos["home/home"]


def test_full_time_results_agree_with_the_match_model() -> None:
    """Summing the halves must reproduce the 1X2 the matrix already gives."""
    from app.ml.dixon_coles import half_time_full_time

    combos = half_time_full_time(1.6, 1.2)
    full = outcomes(score_matrix(1.6, 1.2))
    home = sum(p for key, p in combos.items() if key.endswith("/home"))
    assert home == pytest.approx(full["home_win"], abs=0.02)


def test_team_totals_are_ordered_and_match_the_side() -> None:
    from app.ml.dixon_coles import team_total_over

    matrix = score_matrix(1.9, 0.95)
    assert team_total_over(matrix, 0.5, "home") > team_total_over(matrix, 1.5, "home")
    assert team_total_over(matrix, 0.5, "home") > team_total_over(matrix, 0.5, "away")


def test_odd_and_even_split_the_whole_distribution() -> None:
    from app.ml.dixon_coles import odd_even

    parity = odd_even(score_matrix(1.9, 0.95))
    assert parity["odd"] + parity["even"] == pytest.approx(1.0, abs=1e-9)
    # 0-0 is even, so a low-scoring fixture leans even rather than being a
    # perfect coin flip.
    assert odd_even(score_matrix(0.6, 0.5))["even"] > 0.5
