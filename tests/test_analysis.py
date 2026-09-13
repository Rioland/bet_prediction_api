"""Deep per-fixture analysis: head to head, market coverage and the recommendation."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.models import League, Match, Team
from app.services.analysis import analyse, confidence_band, head_to_head, recommend
from app.services.tips import Tip, build_tips

PREDICTION = {
    "home_win_prob": 0.58, "draw_prob": 0.24, "away_win_prob": 0.18,
    "btts_prob": 0.52, "over_25_prob": 0.57, "home_xg": 1.8, "away_xg": 1.0,
}
CARD = {"home_team": "Arsenal", "away_team": "Brentford",
        "odds_home": 1.75, "odds_draw": 3.8, "odds_away": 5.0}


@pytest.fixture()
def rivals(db_session: Session):
    league = League(external_id=1, name="L", country="C")
    home, away = Team(name="Arsenal"), Team(name="Brentford")
    db_session.add_all([league, home, away])
    db_session.flush()

    base = datetime.utcnow() - timedelta(days=400)
    # Five prior meetings: Arsenal win 3, draw 1, lose 1 (venues alternate).
    results = [(2, 0, True), (1, 1, True), (0, 2, False), (3, 1, True), (1, 0, False)]
    for i, (hg, ag, arsenal_home) in enumerate(results):
        db_session.add(Match(
            external_id=100 + i, league_id=league.id,
            home_team_id=home.id if arsenal_home else away.id,
            away_team_id=away.id if arsenal_home else home.id,
            kickoff_time=base + timedelta(days=30 * i), status="FT", season=2024,
            home_goals=hg, away_goals=ag,
        ))
    upcoming = Match(
        external_id=777, league_id=league.id, home_team_id=home.id, away_team_id=away.id,
        kickoff_time=datetime.utcnow() + timedelta(days=1), status="NS", season=2025,
        odds_home=1.75, odds_draw=3.8, odds_away=5.0,
    )
    db_session.add(upcoming)
    db_session.commit()
    return upcoming, home, away


def test_head_to_head_counts_from_the_home_side_perspective(db_session, rivals) -> None:
    upcoming, home, away = rivals
    h2h = head_to_head(db_session, home.id, away.id)

    assert h2h["played"] == 5
    assert h2h["home_wins"] + h2h["draws"] + h2h["away_wins"] == 5
    # Arsenal: 2-0 W, 1-1 D, 0-2 (away, so Arsenal scored 2) W, 3-1 W, 1-0 (away) L
    assert h2h["home_wins"] == 3
    assert h2h["draws"] == 1
    assert h2h["away_wins"] == 1


def test_head_to_head_reports_goal_context(db_session, rivals) -> None:
    upcoming, home, away = rivals
    h2h = head_to_head(db_session, home.id, away.id)
    assert h2h["avg_goals"] == pytest.approx((2 + 2 + 2 + 4 + 1) / 5)
    assert 0 <= h2h["btts_rate"] <= 1
    assert len(h2h["fixtures"]) == 5


def test_head_to_head_with_no_meetings_is_empty_not_zero(db_session: Session) -> None:
    h2h = head_to_head(db_session, 1, 2)
    assert h2h["played"] == 0
    assert h2h["avg_goals"] is None, "no meetings must not read as 0.0 goals"


def test_recommendation_prefers_edge_over_raw_probability(db_session, rivals) -> None:
    """Near-certain cover bets must not beat a 58% home win at a real price."""
    tips = build_tips(PREDICTION, CARD)
    best = recommend(tips)

    # The highest raw probability is a cover bet - a +1.5 handicap or double
    # chance - which is exactly what the recommendation must not lead with.
    top_by_probability = max(tips, key=lambda t: t.probability)
    assert top_by_probability.selection in {"1X", "1 (+1.5)"}
    assert best["selection"] == "1", f"expected the priced edge, got {best['selection']}"
    assert best["basis"] == "value"


def test_recommendation_falls_back_to_probability_without_odds(db_session) -> None:
    unpriced = {**CARD, "odds_home": None, "odds_draw": None, "odds_away": None}
    best = recommend(build_tips(PREDICTION, unpriced))
    assert best["basis"] == "probability"
    assert "no edge is claimed" in best["why"]


def test_recommendation_is_none_when_nothing_clears_the_floor() -> None:
    flat = {"home_win_prob": 0.34, "draw_prob": 0.33, "away_win_prob": 0.33,
            "btts_prob": 0.40, "over_25_prob": 0.45, "home_xg": 0.6, "away_xg": 0.6}
    # Cover bets and low-goal markets clear the floor almost regardless of the
    # fixture, so they are removed to leave a genuinely flat card.
    excluded = {"double_chance", "under_3_5", "handicap", "first_half"}
    tips = [t for t in build_tips(flat, CARD) if t.market not in excluded]
    assert recommend(tips) is None


def test_confidence_bands_never_imply_certainty() -> None:
    assert confidence_band(0.92)["band"] == "strong"
    assert "fails" in confidence_band(0.92)["note"]
    assert confidence_band(0.58)["band"] == "slight"
    assert confidence_band(0.20)["band"] == "weak"


def test_analysis_covers_every_market_the_user_asked_for(db_session, rivals) -> None:
    upcoming, home, away = rivals
    features = {
        "home_ppg": 2.0, "away_ppg": 1.1, "home_venue_ppg": 2.2, "away_venue_ppg": 0.9,
        "home_goals_scored_avg": 1.9, "away_goals_scored_avg": 1.0,
        "home_goals_conceded_avg": 0.8, "away_goals_conceded_avg": 1.6,
        "home_matches_played": 12, "away_matches_played": 11,
        "home_rest_days": 6.0, "away_rest_days": 4.0,
    }
    result = analyse(db_session, upcoming, features, PREDICTION)

    markets = {m["market"] for m in result["markets"]}
    for expected in ("over_2_5", "over_1_5", "either_half", "double_chance", "btts", "home_win"):
        assert expected in markets, f"missing {expected}"

    selections = {m["selection"] for m in result["markets"]}
    assert {"1WEH", "2WEH", "12", "Over 2.5"} <= selections

    assert result["head_to_head"]["played"] == 5
    assert result["recommendation"]["selection"]
    assert result["expected_goals"]["total"] == pytest.approx(2.8)
    assert result["form"]["home"]["matches_played"] == 12


def test_goal_lines_stay_ordered_in_the_analysis(db_session, rivals) -> None:
    upcoming, _, _ = rivals
    features = {k: 1.2 for k in (
        "home_ppg", "away_ppg", "home_venue_ppg", "away_venue_ppg",
        "home_goals_scored_avg", "away_goals_scored_avg",
        "home_goals_conceded_avg", "away_goals_conceded_avg",
        "home_rest_days", "away_rest_days")}
    features |= {"home_matches_played": 10, "away_matches_played": 10}
    lines = analyse(db_session, upcoming, features, PREDICTION)["goal_lines"]
    assert lines["over_1_5"] >= lines["over_2_5"] >= lines["over_3_5"]


def test_value_pick_surfaces_the_largest_edge_even_below_the_floor() -> None:
    """A 47% shot at 4.06 is the soundest bet on the card and must not vanish."""
    from app.services.analysis import best_value

    longshot = {**CARD, "odds_home": 4.06}
    prediction = {**PREDICTION, "home_win_prob": 0.47, "draw_prob": 0.28, "away_win_prob": 0.25}
    pick = best_value(build_tips(prediction, longshot))

    assert pick["selection"] == "1"
    assert pick["value"] > 0.2
    assert pick["loses_more_often_than_not"] is True
    assert "loses about" in pick["why"]


def test_value_pick_does_not_warn_when_the_pick_is_a_favourite() -> None:
    from app.services.analysis import best_value

    pick = best_value(build_tips(PREDICTION, CARD))
    assert pick["loses_more_often_than_not"] is False
    assert "loses about" not in pick["why"]


def test_value_pick_is_none_without_real_prices() -> None:
    from app.services.analysis import best_value

    unpriced = {**CARD, "odds_home": None, "odds_draw": None, "odds_away": None}
    assert best_value(build_tips(PREDICTION, unpriced)) is None


def test_analysis_declares_which_markets_had_prices(db_session, rivals) -> None:
    upcoming, _, _ = rivals
    features = {k: 1.3 for k in (
        "home_ppg", "away_ppg", "home_venue_ppg", "away_venue_ppg",
        "home_goals_scored_avg", "away_goals_scored_avg",
        "home_goals_conceded_avg", "away_goals_conceded_avg",
        "home_rest_days", "away_rest_days")}
    features |= {"home_matches_played": 10, "away_matches_played": 10}

    result = analyse(db_session, upcoming, features, PREDICTION)
    # The feed carries 1X2 prices only; derived goal markets have none.
    assert "over_2_5" not in result["priced_markets"]
    assert result["value_pick"] is None or result["value_pick"]["market"] in result["priced_markets"]
