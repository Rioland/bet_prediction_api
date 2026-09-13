"""Deep per-fixture analysis.

Scores every market for one fixture, surfaces the evidence behind it - head to
head, recent form, expected goals - and names one recommended selection.

The recommendation is deliberately not "the highest probability". A 90% double
chance priced at 1.05 is a near-certain way to lose money slowly; the pick that
matters is the one whose probability most exceeds what the market charges for
it, among selections the model actually favours. Where no bookmaker price is
available there is no edge to measure, so the fallback ranks on probability and
says so.
"""

from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.ml.dixon_coles import over_line, score_matrix
from app.models import Match
from app.services.tips import MIN_MEANINGFUL_EDGE, MIN_PROBABILITY, Tip, build_tips

# Confidence bands shown to the reader, so "68%" is never presented as a lock.
CONFIDENCE_BANDS = [
    (0.75, "strong", "The model is confident, though roughly one in four still fails."),
    (0.65, "solid", "A clear lean, expected to lose around one time in three."),
    (0.55, "slight", "A marginal edge only - close to a coin flip."),
]


def confidence_band(probability: float) -> dict[str, str]:
    for threshold, label, note in CONFIDENCE_BANDS:
        if probability >= threshold:
            return {"band": label, "note": note}
    return {"band": "weak", "note": "Below the threshold worth acting on."}


def head_to_head(db: Session, home_id: int, away_id: int, limit: int = 10) -> dict[str, Any]:
    """Completed meetings between the two sides, most recent first."""
    stmt = (
        select(Match)
        .where(
            Match.home_goals.is_not(None),
            Match.away_goals.is_not(None),
            or_(
                (Match.home_team_id == home_id) & (Match.away_team_id == away_id),
                (Match.home_team_id == away_id) & (Match.away_team_id == home_id),
            ),
        )
        .order_by(Match.kickoff_time.desc())
        .limit(limit)
    )
    meetings = list(db.scalars(stmt))

    home_wins = away_wins = draws = 0
    total_goals = 0
    both_scored = 0
    fixtures: list[dict[str, Any]] = []

    for match in meetings:
        # Normalise to the perspective of the upcoming home side.
        if match.home_team_id == home_id:
            scored, conceded = match.home_goals, match.away_goals
        else:
            scored, conceded = match.away_goals, match.home_goals

        if scored > conceded:
            home_wins += 1
            result = "home"
        elif scored < conceded:
            away_wins += 1
            result = "away"
        else:
            draws += 1
            result = "draw"

        total_goals += match.home_goals + match.away_goals
        if match.home_goals > 0 and match.away_goals > 0:
            both_scored += 1

        fixtures.append({
            "kickoff": match.kickoff_time.isoformat(),
            "score": f"{match.home_goals}-{match.away_goals}",
            "at_home": match.home_team_id == home_id,
            "result_for_home_side": result,
        })

    played = len(meetings)
    return {
        "played": played,
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "avg_goals": round(total_goals / played, 2) if played else None,
        "btts_rate": round(both_scored / played, 2) if played else None,
        "fixtures": fixtures,
    }


def _form_summary(features: dict[str, float], side: str) -> dict[str, Any]:
    return {
        "points_per_game": round(features[f"{side}_ppg"], 2),
        "goals_scored_avg": round(features[f"{side}_goals_scored_avg"], 2),
        "goals_conceded_avg": round(features[f"{side}_goals_conceded_avg"], 2),
        "matches_played": int(features[f"{side}_matches_played"]),
        "rest_days": round(features[f"{side}_rest_days"], 1),
    }


def recommend(tips: list[Tip]) -> dict[str, Any] | None:
    """Pick one selection to lead with, and say why it was chosen."""
    eligible = [t for t in tips if t.probability >= MIN_PROBABILITY]
    if not eligible:
        return None

    priced = [t for t in eligible if t.value >= MIN_MEANINGFUL_EDGE]
    if priced:
        best = max(priced, key=lambda t: t.value)
        basis = "value"
        why = (
            f"The model rates this at {best.probability:.0%} while the price of "
            f"{best.odds:.2f} implies {1 / best.odds:.0%} - a gap of "
            f"{best.value * 100:.0f} points in your favour."
        )
    else:
        best = max(eligible, key=lambda t: t.probability)
        basis = "probability"
        why = (
            f"No bookmaker price was available to compare against, so this is "
            f"ranked on the model's own {best.probability:.0%} and no edge is claimed."
        )

    return {
        **best.as_dict(),
        "basis": basis,
        "why": why,
        **confidence_band(best.probability),
    }


def best_value(tips: list[Tip]) -> dict[str, Any] | None:
    """The largest edge on the card, regardless of the confidence floor.

    Kept separate from the recommendation on purpose. A 47% shot at 4.06 can be
    the soundest bet available and still lose more often than it wins; leading
    with it would misrepresent it, and hiding it would withhold the most
    interesting price on the card.
    """
    priced = [t for t in tips if t.value >= MIN_MEANINGFUL_EDGE]
    if not priced:
        return None

    best = max(priced, key=lambda t: t.value)
    return {
        **best.as_dict(),
        **confidence_band(best.probability),
        "loses_more_often_than_not": best.probability < 0.5,
        "why": (
            f"The market prices this at {1 / best.odds:.0%} and the model makes it "
            f"{best.probability:.0%}. That is the largest disagreement on the card"
            + (
                f", but it still loses about {1 - best.probability:.0%} of the time."
                if best.probability < 0.5
                else "."
            )
        ),
    }


def analyse(db: Session, match: Match, features: dict[str, float], prediction: dict[str, Any]) -> dict[str, Any]:
    """Full breakdown for one fixture: markets, evidence and a recommendation."""
    card = {
        "home_team": match.home_team.name if match.home_team else "Home",
        "away_team": match.away_team.name if match.away_team else "Away",
        "odds_home": match.odds_home,
        "odds_draw": match.odds_draw,
        "odds_away": match.odds_away,
    }
    tips = build_tips(prediction, card)
    home_xg, away_xg = prediction["home_xg"], prediction["away_xg"]
    matrix = score_matrix(home_xg, away_xg)

    return {
        "fixture_id": match.external_id or match.id,
        "home_team": card["home_team"],
        "away_team": card["away_team"],
        "league_name": match.league.name if match.league else None,
        "kickoff": match.kickoff_time.isoformat(),
        "recommendation": recommend(tips),
        "value_pick": best_value(tips),
        # Only 1X2 prices come from the fixture feed, so edge cannot be measured
        # on the other markets. Saying so beats implying they were all checked.
        "priced_markets": sorted(
            {t.market for t in tips if t.value >= MIN_MEANINGFUL_EDGE or t.value <= -MIN_MEANINGFUL_EDGE}
        ),
        "markets": [t.as_dict() for t in tips],
        "expected_goals": {
            "home": home_xg,
            "away": away_xg,
            "total": round(home_xg + away_xg, 2),
        },
        "goal_lines": {
            "over_1_5": round(over_line(matrix, 1.5), 4),
            "over_2_5": round(over_line(matrix, 2.5), 4),
            "over_3_5": round(over_line(matrix, 3.5), 4),
        },
        "form": {
            "home": _form_summary(features, "home"),
            "away": _form_summary(features, "away"),
        },
        "head_to_head": head_to_head(db, match.home_team_id, match.away_team_id),
    }


# The daily shortlist. Ten is the published card size; the cap leaves headroom
# without running the ranking dry.
DAILY_TARGET = 10
DAILY_MAXIMUM = 15

# When a single day cannot fill the card, the selection reaches into following
# days rather than publishing a short one. Fixture volume swings hard by
# weekday - a Friday with one game sits beside a Saturday with fifty.
LOOKAHEAD_DAYS = 3


def selection_score(pick: dict[str, Any]) -> tuple[float, float]:
    """Rank key for the daily shortlist.

    Selections with a measurable edge come first, ordered by that edge. Where
    no price exists the edge is unknown rather than zero, so those fall back to
    the model's own probability and rank below anything demonstrably priced
    well.
    """
    edge = pick["value"] if pick["value"] >= MIN_MEANINGFUL_EDGE else 0.0
    return (edge, pick["probability"])


def select_daily(
    analysed: list[dict[str, Any]], target: int = DAILY_TARGET, maximum: int = DAILY_MAXIMUM
) -> list[dict[str, Any]]:
    """Choose the day's shortlist, spread across competitions.

    One fixture per league first, so a single busy division cannot take every
    slot, then the strongest remaining until the target is met.
    """
    target = max(1, min(target, maximum))
    ranked = sorted(
        (a for a in analysed if a.get("recommendation")),
        key=lambda a: selection_score(a["recommendation"]),
        reverse=True,
    )

    chosen: list[dict[str, Any]] = []
    seen_leagues: set[Any] = set()
    for entry in ranked:
        league = entry.get("league_name")
        if league in seen_leagues:
            continue
        chosen.append(entry)
        seen_leagues.add(league)
        if len(chosen) == target:
            return chosen

    for entry in ranked:
        if entry not in chosen:
            chosen.append(entry)
            if len(chosen) == target:
                break
    return chosen
