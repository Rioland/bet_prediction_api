"""Market tips, VIP gating and the verified results record."""

from datetime import date as date_type, datetime, time, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from jwt import PyJWTError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import decode_token
from app.database import get_db
from app.ml.features import FEATURE_COLUMNS, features_for_upcoming
from app.ml.dataset import load_finished_matches
from app.models import Match, User
from app.services import results as results_service
from app.services.analysis import (
    DAILY_MAXIMUM,
    DAILY_TARGET,
    LOOKAHEAD_DAYS,
    analyse,
    recommend,
    select_daily,
)
from app.services.prediction_service import predict, predict_many
from app.services.tips import (
    MARKETS,
    build_accumulator,
    build_tips,
    filter_by_market,
    select_banker,
)

router = APIRouter(prefix="/football", tags=["tips"])

FINISHED_STATUSES = {"FT", "AET", "PEN", "FINISHED"}
VIP_ROLES = {"premium_user", "admin", "super_admin"}
VIP_EDGE = 0.04

# Below this, a team's rolling features are mostly neutral priors rather than
# evidence. Predicting from them yields an identical, confident-looking tip for
# every unknown fixture, which is worse than showing nothing.
MIN_TEAM_HISTORY = 5

DbSession = Annotated[Session, Depends(get_db)]


def optional_user(
    db: DbSession, authorization: str | None = Header(default=None)
) -> User | None:
    """Resolve the caller if signed in, without requiring it.

    Tips are browsable anonymously; only the VIP selections are withheld.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        data = decode_token(authorization.split(" ", 1)[1].strip())
        if data.get("type") != "access":
            return None
        return db.get(User, int(data["sub"]))
    except (PyJWTError, ValueError, KeyError):
        return None


def require_user(viewer: Annotated[User | None, Depends(optional_user)]) -> User:
    if viewer is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return viewer


def has_vip(user: User | None) -> bool:
    if user is None:
        return False
    return user.subscription_type == "premium" or user.role in VIP_ROLES


def _matches_on(db: Session, on_date: date_type | None, league_id: int | None) -> list[Match]:
    start = datetime.combine(on_date or datetime.utcnow().date(), time.min)
    stmt = select(Match).where(
        Match.kickoff_time >= start, Match.kickoff_time < start + timedelta(days=1)
    )
    if league_id is not None:
        stmt = stmt.where(Match.league_id == league_id)
    return list(db.scalars(stmt.order_by(Match.kickoff_time)))


def _features_for(db: Session, matches: list[Match]) -> dict[int, dict[str, float]]:
    """Leak-free feature rows for the given fixtures, keyed by match id."""
    if not matches:
        return {}

    history = load_finished_matches(db)
    upcoming = [
        {
            "id": m.id,
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "kickoff_time": m.kickoff_time,
            "season": m.season,
        }
        for m in matches
    ]
    frame = features_for_upcoming(history, upcoming)
    if frame.empty:
        return {}

    return {
        int(row["match_id"]): {c: float(row[c]) for c in FEATURE_COLUMNS}
        for _, row in frame.iterrows()
    }


def _predict_from(features: dict[str, float]) -> dict[str, float]:
    """Calibrated model output plus the expected goals the derived markets need."""
    winner = predict("match_winner", features)
    btts = predict("btts", features)
    totals = predict("over_under_2_5", features)
    probs = winner["probabilities"]
    home_xg = (features["home_goals_scored_avg"] + features["away_goals_conceded_avg"]) / 2 * 1.08
    away_xg = (features["away_goals_scored_avg"] + features["home_goals_conceded_avg"]) / 2 * 0.94
    return {
        "home_win_prob": probs.get("H", 0.0),
        "draw_prob": probs.get("D", 0.0),
        "away_win_prob": probs.get("A", 0.0),
        "btts_prob": btts["probabilities"].get("yes", 0.0),
        "over_25_prob": totals["probabilities"].get("over", 0.0),
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
    }


def has_enough_history(features: dict[str, float]) -> bool:
    return min(features["home_matches_played"], features["away_matches_played"]) >= MIN_TEAM_HISTORY


def _tip_cards(db: Session, matches: list[Match], market: str) -> list[dict]:
    feature_rows = _features_for(db, matches)
    cards: list[dict] = []
    for match in matches:
        features = feature_rows.get(match.id)
        # Newly ingested teams carry no history; skip rather than dress a prior
        # up as a prediction.
        if features is None or not has_enough_history(features):
            continue

        prediction = _predict_from(features)
        card = {
            "fixture_id": match.external_id or match.id,
            "league_id": match.league_id,
            "league_name": match.league.name if match.league else "Unknown",
            "league_country": match.league.country if match.league else None,
            "home_team": match.home_team.name if match.home_team else "Unknown",
            "home_logo": match.home_team.logo_url if match.home_team else None,
            "away_team": match.away_team.name if match.away_team else "Unknown",
            "away_logo": match.away_team.logo_url if match.away_team else None,
            "kickoff": match.kickoff_time.isoformat(),
            "status": match.status,
            "odds_home": match.odds_home,
            "odds_draw": match.odds_draw,
            "odds_away": match.odds_away,
        }
        candidates = filter_by_market(build_tips(prediction, card), market)
        card["tip"] = candidates[0].as_dict() if candidates else None
        card["all_tips"] = [t.as_dict() for t in candidates]
        cards.append(card)
    return cards


@router.get("/markets")
def list_markets() -> list[str]:
    return MARKETS


@router.get("/tips")
def tips(
    db: DbSession,
    viewer: Annotated[User | None, Depends(optional_user)],
    on_date: date_type | None = Query(default=None, alias="date"),
    market: str = Query(default="popular"),
    league_id: int | None = Query(default=None),
) -> dict:
    """Tip cards for a date and market.

    The VIP count is disclosed while the picks are withheld, so the paywall is
    visible rather than the page merely looking empty.
    """
    if market not in MARKETS:
        raise HTTPException(status_code=422, detail=f"Unknown market '{market}'")

    fixtures = [m for m in _matches_on(db, on_date, league_id) if m.status not in FINISHED_STATUSES]
    cards = [c for c in _tip_cards(db, fixtures, market) if c["tip"]]
    cards.sort(key=lambda c: c["tip"]["probability"], reverse=True)

    vip_cards = [c for c in cards if c["tip"]["value"] >= VIP_EDGE][:3]
    vip_ids = {c["fixture_id"] for c in vip_cards}
    free_cards = [c for c in cards if c["fixture_id"] not in vip_ids]

    unlocked = has_vip(viewer)
    payload: dict = {
        "date": (on_date or datetime.utcnow().date()).isoformat(),
        "market": market,
        "matches": free_cards,
        "vip_locked": not unlocked,
        "vip_count": len(vip_cards),
    }
    if unlocked:
        payload["vip_matches"] = vip_cards
    if market == "banker":
        payload["banker"] = select_banker(free_cards)
    if market in ("2_odds", "acca"):
        payload["accumulator"] = build_accumulator(
            free_cards, target_odds=2.0 if market == "2_odds" else 5.0
        )
    return payload


@router.get("/vip/tips")
def vip_tips(
    db: DbSession,
    current_user: Annotated[User, Depends(require_user)],
    on_date: date_type | None = Query(default=None, alias="date"),
    market: str = Query(default="popular"),
) -> dict:
    if not has_vip(current_user):
        raise HTTPException(status_code=402, detail="VIP subscription required")

    fixtures = [m for m in _matches_on(db, on_date, None) if m.status not in FINISHED_STATUSES]
    cards = [c for c in _tip_cards(db, fixtures, market) if c["tip"]]
    cards.sort(key=lambda c: c["tip"]["value"], reverse=True)
    return {"date": (on_date or datetime.utcnow().date()).isoformat(), "matches": cards}


@router.get("/results/performance")
def tracked_performance(db: DbSession, days: int = Query(default=30, ge=1, le=365)) -> dict:
    """Real settled performance, from tips recorded before kickoff."""
    return {
        "overall": results_service.performance(db, days=days),
        "by_market": results_service.performance_by_market(db, days=days),
    }


@router.get("/results/recent")
def recent_results(db: DbSession, limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    return [
        {
            "match_id": t.match_id,
            "home_team": t.match.home_team.name if t.match and t.match.home_team else None,
            "away_team": t.match.away_team.name if t.match and t.match.away_team else None,
            "kickoff": t.kickoff_time.isoformat(),
            "market": t.market,
            "selection": t.selection,
            "odds": t.odds,
            "probability": t.probability,
            "result": t.result,
            "score": (
                f"{t.match.home_goals}-{t.match.away_goals}"
                if t.match and t.match.home_goals is not None
                else None
            ),
        }
        for t in results_service.recent_settled(db, limit=limit)
    ]


@router.get("/analysis/{fixture_id}")
def fixture_analysis(fixture_id: int, db: DbSession) -> dict:
    """Every market for one fixture, the evidence behind it, and a recommendation.

    The recommendation is the selection whose probability most exceeds its
    market price, not the highest probability outright: a 90% double chance at
    1.05 is a reliable way to lose money slowly.
    """
    match = db.scalar(select(Match).where(Match.external_id == fixture_id)) or db.get(
        Match, fixture_id
    )
    if match is None:
        raise HTTPException(status_code=404, detail="Fixture not found")

    features = _features_for(db, [match]).get(match.id)
    if features is None or not has_enough_history(features):
        home_played = int(features["home_matches_played"]) if features else 0
        away_played = int(features["away_matches_played"]) if features else 0
        home = match.home_team.name if match.home_team else "the home side"
        away = match.away_team.name if match.away_team else "the away side"
        raise HTTPException(
            status_code=409,
            detail=(
                f"{home} has {home_played} completed matches on record and {away} has "
                f"{away_played}; {MIN_TEAM_HISTORY} each are needed. Results are "
                "collected as fixtures are played, so this fills in over time."
            ),
        )

    return analyse(db, match, features, _predict_from(features))


@router.get("/fixtures")
def all_fixtures(
    db: DbSession,
    on_date: date_type | None = Query(default=None, alias="date"),
    league_id: int | None = Query(default=None),
) -> dict:
    """Every fixture on a date, analysable or not.

    Fixtures the model cannot speak to are still listed, with the reason,
    rather than silently dropped: an incomplete card looks like missing
    fixtures, not like a model declining to guess.
    """
    matches = _matches_on(db, on_date, league_id)
    feature_rows = _features_for(db, matches)

    fixtures: list[dict] = []
    analysable = 0
    for match in matches:
        features = feature_rows.get(match.id)
        entry = {
            "fixture_id": match.external_id or match.id,
            "league_id": match.league_id,
            "league_name": match.league.name if match.league else "Unknown",
            "league_country": match.league.country if match.league else None,
            "home_team": match.home_team.name if match.home_team else "Unknown",
            "home_logo": match.home_team.logo_url if match.home_team else None,
            "away_team": match.away_team.name if match.away_team else "Unknown",
            "away_logo": match.away_team.logo_url if match.away_team else None,
            "kickoff": match.kickoff_time.isoformat(),
            "status": match.status,
            "home_score": match.home_goals,
            "away_score": match.away_goals,
            "odds_home": match.odds_home,
            "odds_draw": match.odds_draw,
            "odds_away": match.odds_away,
            "tip": None,
            "analysis_available": False,
            "unavailable_reason": None,
        }

        if features is None:
            entry["unavailable_reason"] = "No feature data for this fixture."
        elif not has_enough_history(features):
            played = int(min(features["home_matches_played"], features["away_matches_played"]))
            entry["unavailable_reason"] = (
                f"Only {played} completed matches on record for one of these sides; "
                f"{MIN_TEAM_HISTORY} are needed."
            )
        else:
            candidates = filter_by_market(build_tips(_predict_from(features), entry), "popular")
            entry["analysis_available"] = True
            entry["tip"] = candidates[0].as_dict() if candidates else None
            if not candidates:
                entry["unavailable_reason"] = "No selection cleared the confidence floor."
            analysable += 1

        fixtures.append(entry)

    return {
        "date": (on_date or datetime.utcnow().date()).isoformat(),
        "total": len(fixtures),
        "analysable": analysable,
        "fixtures": fixtures,
    }


def _candidates_from(db: Session, start: date_type, days: int) -> list[tuple[Match, dict]]:
    """Upcoming fixtures with usable history, across a span of days."""
    matches: list[Match] = []
    for offset in range(days):
        matches.extend(
            m
            for m in _matches_on(db, start + timedelta(days=offset), None)
            if m.status not in FINISHED_STATUSES
        )
    feature_rows = _features_for(db, matches)
    return [
        (match, features)
        for match in matches
        if (features := feature_rows.get(match.id)) is not None and has_enough_history(features)
    ]


def select_for_date(
    db: Session, start: date_type, limit: int = DAILY_TARGET
) -> dict:
    """The day's shortlist, each selected fixture fully analysed.

    Candidates are ranked from a cheap pass over features and model output,
    and only the fixtures that make the card get the full breakdown. Analysing
    everything first and discarding most of it took roughly ten seconds on a
    fifty-fixture day.

    If the requested date cannot fill the card the window extends into the
    following days, since fixture volume swings hard by weekday.
    """
    candidates = _candidates_from(db, start, 1)
    days_used = 1

    # Reach forward only when the day itself is short.
    while len(candidates) < limit and days_used <= LOOKAHEAD_DAYS:
        days_used += 1
        candidates = _candidates_from(db, start, days_used)

    # One batched pass per target instead of three calls per fixture.
    feature_rows = [features for _, features in candidates]
    winners = predict_many("match_winner", feature_rows)
    btts_all = predict_many("btts", feature_rows)
    totals_all = predict_many("over_under_2_5", feature_rows)

    ranked: list[dict] = []
    for index, (match, features) in enumerate(candidates):
        probs = winners[index]["probabilities"]
        home_xg = (features["home_goals_scored_avg"] + features["away_goals_conceded_avg"]) / 2 * 1.08
        away_xg = (features["away_goals_scored_avg"] + features["home_goals_conceded_avg"]) / 2 * 0.94
        prediction = {
            "home_win_prob": probs.get("H", 0.0),
            "draw_prob": probs.get("D", 0.0),
            "away_win_prob": probs.get("A", 0.0),
            "btts_prob": btts_all[index]["probabilities"].get("yes", 0.0),
            "over_25_prob": totals_all[index]["probabilities"].get("over", 0.0),
            "home_xg": round(home_xg, 2),
            "away_xg": round(away_xg, 2),
        }
        card = {
            "home_team": match.home_team.name if match.home_team else "Home",
            "away_team": match.away_team.name if match.away_team else "Away",
            "odds_home": match.odds_home,
            "odds_draw": match.odds_draw,
            "odds_away": match.odds_away,
        }
        recommendation = recommend(build_tips(prediction, card))
        if recommendation is None:
            continue
        ranked.append({
            "match": match,
            "features": features,
            "prediction": prediction,
            "league_name": match.league.name if match.league else None,
            "recommendation": recommendation,
        })

    shortlist = select_daily(ranked, target=limit)
    selected = [
        analyse(db, entry["match"], entry["features"], entry["prediction"])
        for entry in shortlist
    ]

    # Every ranked fixture in leg shape. Slips draw on this rather than the
    # shortlist: a market-themed slip needs several legs of one market, which
    # a league-diverse top ten rarely contains.
    pool = [
        {
            "fixture_id": entry["match"].external_id or entry["match"].id,
            "home_team": entry["match"].home_team.name if entry["match"].home_team else "Home",
            "away_team": entry["match"].away_team.name if entry["match"].away_team else "Away",
            "league_name": entry["league_name"],
            "kickoff": entry["match"].kickoff_time.isoformat(),
            "recommendation": entry["recommendation"],
        }
        for entry in ranked
    ]

    return {
        "date": start.isoformat(),
        "days_covered": days_used,
        "considered": len(candidates),
        "analysed": len(ranked),
        "selected": len(selected),
        "matches": selected,
        "pool": pool,
    }


@router.get("/daily-selection")
def daily_selection(
    db: DbSession,
    on_date: date_type | None = Query(default=None, alias="date"),
    limit: int = Query(default=DAILY_TARGET, ge=1, le=DAILY_MAXIMUM),
) -> dict:
    """The day's shortlist, each selected fixture fully analysed."""
    result = select_for_date(db, on_date or datetime.utcnow().date(), limit)
    # Internal only: the slip builder needs it, API consumers do not.
    result.pop("pool", None)
    return result
