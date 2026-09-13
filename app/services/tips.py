"""Turn calibrated model output into per-market tip selections.

Markets map to models like this:

* 1X2, Double Chance      - directly from the match_winner model
* BTTS                    - directly from the btts model
* Over/Under 2.5          - directly from the over_under_2_5 model
* Other goal lines (1.5, 3.5) - DERIVED, not separately trained: total goals are
  modelled as Poisson around the expected-goals estimate. These are labelled
  ``derived`` so the UI can be honest that they are an inference, not a fit.

Every tip carries the probability it was selected on. ``value`` is the gap
between that probability and the odds-implied one; it is the only figure here
that says anything about whether a bet is worth placing.
"""

from dataclasses import dataclass
from typing import Any

from app.ml.dixon_coles import over_line, score_matrix

# A tip is only surfaced if the model is at least this sure.
MIN_PROBABILITY = 0.55
# VIP picks are the ones that also show positive expected value.
VIP_MIN_VALUE = 0.04

MARKETS = [
    "popular", "banker", "2_odds", "home_win", "away_win", "draws",
    "double_chance", "btts", "over_1_5", "over_2_5", "under_3_5", "acca",
]


@dataclass
class Tip:
    market: str
    selection: str
    label: str
    probability: float
    odds: float
    fair_odds: float
    value: float
    rationale: str
    source: str  # "model" or "derived"

    def as_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "selection": self.selection,
            "label": self.label,
            "probability": round(self.probability, 4),
            "odds": round(self.odds, 2),
            "fair_odds": round(self.fair_odds, 2),
            "value": round(self.value, 4),
            "rationale": self.rationale,
            "source": self.source,
        }


def total_goals_over(line: float, home_xg: float, away_xg: float) -> float:
    """P(total goals > line) from the Dixon-Coles scoreline distribution.

    Reading every goal line off one joint distribution keeps them mutually
    consistent: P(over 1.5) can never come out below P(over 2.5), which
    deriving each line separately does not guarantee.
    """
    return over_line(score_matrix(home_xg, away_xg), line)


def _fair(probability: float) -> float:
    return 1 / probability if probability > 0 else 999.0


def _tip(market, selection, label, probability, offered_odds, rationale, source="model") -> Tip:
    fair = _fair(probability)
    # With no market price, quote fair odds and claim no value.
    odds = offered_odds if offered_odds and offered_odds > 1 else fair
    implied = 1 / odds if odds > 0 else 1.0
    return Tip(market, selection, label, probability, odds, fair,
               probability - implied, rationale, source)


def build_tips(prediction: dict[str, Any], match: dict[str, Any]) -> list[Tip]:
    """Every candidate tip for one fixture, strongest first."""
    home = match.get("home_team", "Home")
    away = match.get("away_team", "Away")
    p_home = prediction["home_win_prob"]
    p_draw = prediction["draw_prob"]
    p_away = prediction["away_win_prob"]
    home_xg, away_xg = prediction["home_xg"], prediction["away_xg"]
    expected_total = home_xg + away_xg

    tips: list[Tip] = [
        _tip("home_win", "1", f"{home} to win", p_home, match.get("odds_home"),
             f"{home} rate {p_home:.0%} on form and matchup strength."),
        _tip("away_win", "2", f"{away} to win", p_away, match.get("odds_away"),
             f"{away} rate {p_away:.0%} on form and matchup strength."),
        _tip("draws", "X", "Draw", p_draw, match.get("odds_draw"),
             f"Both sides rate within a point of each other - a {p_draw:.0%} draw shout."),
        _tip("double_chance", "1X", f"{home} or draw", p_home + p_draw, None,
             f"{home} avoid defeat in {p_home + p_draw:.0%} of modelled outcomes."),
        _tip("double_chance", "X2", f"{away} or draw", p_away + p_draw, None,
             f"{away} avoid defeat in {p_away + p_draw:.0%} of modelled outcomes."),
        _tip("btts", "GG", "Both teams to score", prediction["btts_prob"], None,
             f"Both sides score in {prediction['btts_prob']:.0%} of modelled outcomes."),
        _tip("over_2_5", "Over 2.5", "Over 2.5 goals", prediction["over_25_prob"], None,
             f"Goals model puts this at {prediction['over_25_prob']:.0%} for over 2.5."),
        _tip("over_1_5", "Over 1.5", "Over 1.5 goals",
             total_goals_over(1.5, home_xg, away_xg), None,
             f"Expected goals of {expected_total:.1f} across the fixture.", source="derived"),
        _tip("under_3_5", "Under 3.5", "Under 3.5 goals",
             1 - total_goals_over(3.5, home_xg, away_xg), None,
             f"Expected goals of {expected_total:.1f} keeps this below 3.5.", source="derived"),
    ]
    return sorted(tips, key=lambda t: t.probability, reverse=True)


def best_tip(prediction: dict[str, Any], match: dict[str, Any]) -> Tip | None:
    """The single strongest selection for a fixture, if any clears the floor."""
    candidates = [t for t in build_tips(prediction, match) if t.probability >= MIN_PROBABILITY]
    return candidates[0] if candidates else None


# Double chance is close to tautological - "1X" is by construction at least as
# likely as "1" - so it dominates any probability ranking and would be the only
# thing ever shown. It stays available as its own tab, but is kept out of the
# mixed views.
_EXCLUDED_FROM_MIXED = {"double_chance"}


def filter_by_market(tips: list[Tip], market: str) -> list[Tip]:
    if market in ("popular", "acca", "banker", "2_odds"):
        mixed = [
            t for t in tips
            if t.market not in _EXCLUDED_FROM_MIXED and t.probability >= MIN_PROBABILITY
        ]
        # Among selections the model actually favours, lead with the one that
        # most beats its market price. Ranking on value alone would surface
        # 25% longshots as headline tips just because the odds were generous.
        priced = [t for t in mixed if t.value > 0]
        return sorted(priced or mixed, key=lambda t: (t.value, t.probability), reverse=True)
    # An explicit market tab still must not present a coin-flip as a call: a
    # 46% BTTS "tip" recommends an outcome the model rates as unlikely.
    return [t for t in tips if t.market == market and t.probability >= MIN_PROBABILITY]


def select_banker(cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Highest-probability single selection across the whole card."""
    scored = [c for c in cards if c.get("tip")]
    if not scored:
        return None
    return max(scored, key=lambda c: c["tip"]["probability"])


def build_accumulator(
    cards: list[dict[str, Any]], target_odds: float = 2.0, max_legs: int = 5
) -> dict[str, Any]:
    """Greedily combine the safest selections until the target price is reached.

    Accumulator probability is the product of the legs, which falls away fast:
    five 80% legs is a 33% chance of returning anything. That combined figure is
    reported so the risk is visible rather than implied by the odds alone.
    """
    ordered = sorted(
        (c for c in cards if c.get("tip")),
        key=lambda c: c["tip"]["probability"],
        reverse=True,
    )
    legs: list[dict[str, Any]] = []
    combined_odds = 1.0
    combined_probability = 1.0

    for card in ordered:
        if combined_odds >= target_odds or len(legs) >= max_legs:
            break
        tip = card["tip"]
        legs.append({
            "fixture_id": card["fixture_id"],
            "home_team": card["home_team"],
            "away_team": card["away_team"],
            "kickoff": card["kickoff"],
            "selection": tip["selection"],
            "label": tip["label"],
            "odds": tip["odds"],
            "probability": tip["probability"],
        })
        combined_odds *= tip["odds"]
        combined_probability *= tip["probability"]

    return {
        "legs": legs,
        "total_odds": round(combined_odds, 2),
        "combined_probability": round(combined_probability, 4),
        "target_odds": target_odds,
    }
