"""Tip generation, settlement and tracked-performance tests."""

import pytest

from app.services.results import settle_selection
from app.services.tips import build_accumulator, build_tips, total_goals_over

PREDICTION = {
    "home_win_prob": 0.62, "draw_prob": 0.23, "away_win_prob": 0.15,
    "btts_prob": 0.48, "over_25_prob": 0.55,
    "predicted_winner": "home", "confidence": 0.62,
    "home_xg": 1.8, "away_xg": 0.9, "predicted_score": "2-1",
}
MATCH = {"home_team": "Arsenal", "away_team": "Brentford",
         "odds_home": 1.70, "odds_draw": 3.90, "odds_away": 5.50}


def test_goal_lines_are_ordered_and_bounded() -> None:
    for home_xg, away_xg in ((0.7, 0.5), (1.5, 1.2), (2.2, 1.8)):
        p15 = total_goals_over(1.5, home_xg, away_xg)
        p25 = total_goals_over(2.5, home_xg, away_xg)
        p35 = total_goals_over(3.5, home_xg, away_xg)
        assert 0 <= p35 <= p25 <= p15 <= 1, f"lines out of order at {home_xg}/{away_xg}"


def test_average_match_prices_over_25_near_the_real_world_rate() -> None:
    # A 2.7-goal fixture should sit close to the ~50% over-2.5 rate football shows.
    assert 0.45 < total_goals_over(2.5, 1.5, 1.2) < 0.60


def test_tips_cover_every_market_and_sort_by_probability() -> None:
    tips = build_tips(PREDICTION, MATCH)
    markets = {t.market for t in tips}
    assert {"home_win", "away_win", "draws", "double_chance", "btts",
            "over_2_5", "over_1_5", "under_3_5"} <= markets

    probabilities = [t.probability for t in tips]
    assert probabilities == sorted(probabilities, reverse=True)


def test_double_chance_probability_is_the_sum_of_its_parts() -> None:
    tips = {(t.market, t.selection): t for t in build_tips(PREDICTION, MATCH)}
    assert tips[("double_chance", "1X")].probability == pytest.approx(0.85)
    assert tips[("double_chance", "X2")].probability == pytest.approx(0.38)


def test_value_is_measured_against_the_offered_price() -> None:
    tips = {(t.market, t.selection): t for t in build_tips(PREDICTION, MATCH)}
    home = tips[("home_win", "1")]
    # 0.62 model probability vs 1/1.70 = 0.588 implied.
    assert home.value == pytest.approx(0.62 - 1 / 1.70, abs=1e-6)


def test_missing_odds_fall_back_to_fair_price_with_no_claimed_value() -> None:
    tips = {(t.market, t.selection): t for t in build_tips(PREDICTION, {**MATCH, "odds_home": None})}
    home = tips[("home_win", "1")]
    assert home.odds == pytest.approx(1 / 0.62, abs=1e-6)
    assert home.value == pytest.approx(0.0, abs=1e-6)


def test_derived_goal_lines_are_labelled_as_derived() -> None:
    tips = {(t.market, t.selection): t for t in build_tips(PREDICTION, MATCH)}
    assert tips[("over_1_5", "Over 1.5")].source == "derived"
    assert tips[("over_2_5", "Over 2.5")].source == "model"


def test_accumulator_reports_compounding_risk() -> None:
    cards = [
        {"fixture_id": i, "home_team": f"H{i}", "away_team": f"A{i}", "kickoff": "2026-01-01T15:00:00",
         "tip": {"selection": "1", "label": "x", "odds": 1.3, "probability": 0.8}}
        for i in range(5)
    ]
    acca = build_accumulator(cards, target_odds=2.0)
    assert acca["total_odds"] >= 2.0
    # Three 0.8 legs is 0.512 - materially worse than any single leg.
    assert acca["combined_probability"] < 0.8
    assert acca["combined_probability"] == pytest.approx(0.8 ** len(acca["legs"]), abs=1e-6)


@pytest.mark.parametrize(
    "market,selection,hg,ag,expected",
    [
        ("home_win", "1", 2, 0, "won"),
        ("home_win", "1", 1, 1, "lost"),
        ("away_win", "2", 0, 3, "won"),
        ("draws", "X", 1, 1, "won"),
        ("double_chance", "1X", 1, 1, "won"),
        ("double_chance", "1X", 0, 1, "lost"),
        ("double_chance", "X2", 0, 0, "won"),
        ("btts", "GG", 1, 1, "won"),
        ("btts", "GG", 3, 0, "lost"),
        ("over_1_5", "Over 1.5", 1, 1, "won"),
        ("over_1_5", "Over 1.5", 1, 0, "lost"),
        ("over_2_5", "Over 2.5", 2, 1, "won"),
        ("under_3_5", "Under 3.5", 2, 1, "won"),
        ("under_3_5", "Under 3.5", 2, 2, "lost"),
        ("unknown_market", "?", 1, 0, "void"),
    ],
)
def test_settlement_rules(market, selection, hg, ag, expected) -> None:
    assert settle_selection(market, selection, hg, ag) == expected


def test_mixed_views_exclude_double_chance() -> None:
    """1X is by construction >= 1, so it would win every probability ranking."""
    from app.services.tips import filter_by_market

    tips = build_tips(PREDICTION, MATCH)
    for market in ("popular", "banker", "2_odds", "acca"):
        assert all(t.market != "double_chance" for t in filter_by_market(tips, market)), market


def test_double_chance_is_still_available_on_its_own_tab() -> None:
    from app.services.tips import filter_by_market

    selections = {t.selection for t in filter_by_market(build_tips(PREDICTION, MATCH), "double_chance")}
    # 1X (0.85) and 12 (0.77) clear the floor; X2 (0.38) is correctly withheld.
    assert selections == {"1X", "12"}


def test_mixed_views_prefer_a_priced_selection_with_edge() -> None:
    from app.services.tips import filter_by_market

    top = filter_by_market(build_tips(PREDICTION, MATCH), "popular")[0]
    assert top.value > 0, "a mixed view should lead with a selection that beats its price"


def test_mixed_views_never_headline_a_longshot() -> None:
    """Generous odds must not promote a selection the model does not favour."""
    from app.services.tips import MIN_PROBABILITY, filter_by_market

    longshot_match = {**MATCH, "odds_away": 12.0}  # huge implied edge on a 15% pick
    for tip in filter_by_market(build_tips(PREDICTION, longshot_match), "popular"):
        assert tip.probability >= MIN_PROBABILITY


def test_market_tabs_do_not_present_unlikely_outcomes_as_tips() -> None:
    """A market tab returning a sub-50% selection would be recommending a loser."""
    from app.services.tips import MIN_PROBABILITY, filter_by_market

    # BTTS at 46% - the model rates it against.
    unlikely = {**PREDICTION, "btts_prob": 0.46}
    assert filter_by_market(build_tips(unlikely, MATCH), "btts") == []

    likely = {**PREDICTION, "btts_prob": 0.72}
    btts = filter_by_market(build_tips(likely, MATCH), "btts")
    assert btts and all(t.probability >= MIN_PROBABILITY for t in btts)


def test_unpriced_selections_claim_no_edge() -> None:
    """Fair odds leave value at zero; floating point returns ~1e-16, not 0."""
    from app.services.tips import MIN_MEANINGFUL_EDGE

    unpriced = {**MATCH, "odds_home": None, "odds_draw": None, "odds_away": None}
    for tip in build_tips(PREDICTION, unpriced):
        assert tip.value < MIN_MEANINGFUL_EDGE, f"{tip.selection} claimed edge with no price"


def test_win_either_half_beats_the_outright_win() -> None:
    """Two chances to win one half must be likelier than winning the match."""
    tips = {(t.market, t.selection): t for t in build_tips(PREDICTION, MATCH)}
    assert tips[("either_half", "1WEH")].probability > tips[("home_win", "1")].probability
    assert tips[("either_half", "2WEH")].probability > tips[("away_win", "2")].probability


def test_first_half_and_handicap_markets_are_offered() -> None:
    markets = {t.market for t in build_tips(PREDICTION, MATCH)}
    assert {"first_half", "first_half_goals", "handicap"} <= markets


def test_first_half_selections_are_labelled_derived() -> None:
    """No model is trained on half-time results; they come from the scoreline model."""
    for tip in build_tips(PREDICTION, MATCH):
        if tip.market in ("first_half", "first_half_goals", "handicap"):
            assert tip.source == "derived", tip.selection


def test_handicap_and_outright_stay_consistent() -> None:
    """Winning by two or more must be rarer than simply winning."""
    tips = {(t.market, t.selection): t for t in build_tips(PREDICTION, MATCH)}
    assert tips[("handicap", "1 (-1.5)")].probability < tips[("home_win", "1")].probability
    assert tips[("handicap", "1 (+1.5)")].probability > tips[("home_win", "1")].probability


def test_cover_bets_are_kept_out_of_mixed_views() -> None:
    """A +1.5 handicap wins unless a side loses by two, so it would top every
    probability ranking exactly as double chance does."""
    from app.services.tips import filter_by_market

    for market in ("popular", "banker", "2_odds", "acca"):
        for tip in filter_by_market(build_tips(PREDICTION, MATCH), market):
            assert "(+" not in tip.selection, f"{tip.selection} surfaced in {market}"
            assert tip.market != "double_chance"


def test_cover_bets_remain_on_their_own_tab() -> None:
    from app.services.tips import filter_by_market

    selections = {t.selection for t in filter_by_market(build_tips(PREDICTION, MATCH), "handicap")}
    assert any("(+" in s for s in selections), "the handicap tab should still offer them"


def test_new_markets_are_offered_and_named_for_the_teams() -> None:
    tips = build_tips(PREDICTION, MATCH)
    markets = {t.market for t in tips}
    assert {"anytime_lead", "win_to_nil", "clean_sheet", "ht_ft",
            "team_goals", "odd_even", "handicap", "btts"} <= markets

    lead = next(t for t in tips if t.market == "anytime_lead" and t.selection == "1 LEAD")
    assert lead.label == "Arsenal to lead at any point"
    assert lead.source == "derived", "in-play leading is inferred, not fitted"


def test_every_handicap_line_is_quoted_for_both_sides() -> None:
    lines = {t.selection for t in build_tips(PREDICTION, MATCH) if t.market == "handicap"}
    assert lines == {
        f"{side} ({line:+.1f})" for side in ("1", "2")
        for line in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
    }


def test_handicap_tips_agree_with_the_model() -> None:
    from app.ml.dixon_coles import handicap

    tip = next(
        t for t in build_tips(PREDICTION, MATCH)
        if t.market == "handicap" and t.selection == "1 (-1.5)"
    )
    assert tip.probability == pytest.approx(handicap(1.8, 0.9, -1.5, "home"))


def test_half_time_full_time_offers_all_nine_doubles() -> None:
    tips = [t for t in build_tips(PREDICTION, MATCH) if t.market == "ht_ft"]
    assert len(tips) == 9
    assert sum(t.probability for t in tips) == pytest.approx(1.0, abs=1e-9)
    comeback = next(t for t in tips if t.selection == "2/1")
    assert comeback.label == "Brentford ahead at half time, Arsenal win"


def test_near_certain_selections_stay_out_of_the_popular_card() -> None:
    """A side merely scoring, or a +2.5 start, would top every mixed ranking."""
    from app.services.tips import filter_by_market

    popular = {t.selection for t in filter_by_market(build_tips(PREDICTION, MATCH), "popular")}
    assert "1 Over 0.5" not in popular
    assert "1 (+2.5)" not in popular
    assert "1 LEAD" not in popular


def test_a_coin_flip_market_is_reachable_on_its_own_tab() -> None:
    """Excluded from mixed views is not the same as unavailable."""
    from app.services.tips import MARKETS, filter_by_market

    assert "odd_even" in MARKETS
    tips = build_tips({**PREDICTION, "home_xg": 0.6, "away_xg": 0.5}, MATCH)
    assert filter_by_market(tips, "odd_even"), "the tab must still return its selection"


@pytest.mark.parametrize(
    ("market", "selection", "score", "expected"),
    [
        ("handicap", "1 (-1.5)", (3, 1), "won"),
        ("handicap", "1 (-1.5)", (2, 1), "lost"),
        ("handicap", "2 (+1.5)", (2, 1), "won"),
        ("handicap", "2 (+1.5)", (3, 1), "lost"),
        ("handicap", "1 (-0.5)", (1, 0), "won"),
        ("win_to_nil", "1 WTN", (2, 0), "won"),
        ("win_to_nil", "1 WTN", (2, 1), "lost"),
        ("clean_sheet", "1 CS", (0, 0), "won"),
        ("clean_sheet", "2 CS", (0, 0), "won"),
        ("team_goals", "1 Over 1.5", (2, 0), "won"),
        ("team_goals", "1 Over 1.5", (1, 3), "lost"),
        ("odd_even", "Odd", (2, 1), "won"),
        ("odd_even", "Even", (0, 0), "won"),
        ("double_chance", "12", (1, 0), "won"),
        ("double_chance", "12", (1, 1), "lost"),
    ],
)
def test_new_markets_settle_from_the_final_score(market, selection, score, expected) -> None:
    assert settle_selection(market, selection, *score) == expected


@pytest.mark.parametrize(
    ("market", "selection"),
    [
        ("anytime_lead", "1 LEAD"),
        ("ht_ft", "1/1"),
        ("either_half", "1WEH"),
        ("first_half", "1H1"),
    ],
)
def test_in_play_markets_are_voided_rather_than_guessed(market, selection) -> None:
    """A final score cannot say who led at the break, so no record is claimed."""
    assert settle_selection(market, selection, 3, 0) == "void"


def test_the_nine_way_market_is_judged_on_its_own_scale() -> None:
    """A 55% gate written for two-way markets would empty this tab every day."""
    from app.services.tips import filter_by_market

    tips = build_tips(PREDICTION, MATCH)
    offered = filter_by_market(tips, "ht_ft")
    assert offered, "the strongest half time/full time double must be reachable"
    assert offered[0].probability < 0.55


def test_a_weak_nine_way_call_never_reaches_the_popular_card() -> None:
    """Reachable on its own tab is not the same as fit to lead the card."""
    from app.services.tips import filter_by_market

    popular = filter_by_market(build_tips(PREDICTION, MATCH), "popular")
    assert all(t.market != "ht_ft" for t in popular)
