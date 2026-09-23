"""Settle published tips against real results and report tracked performance.

Nothing here invents a record: a tip must have been written to published_tips
before kickoff to be counted, and settlement reads only the final score. That
is what separates a verifiable track record from the decorative "winning slips"
that prediction sites usually show.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Match, PublishedTip
from app.services.ingest import FINISHED_STATUSES


# Selections written as a side and a number: "1 (-1.5)" for a handicap,
# "2 Over 0.5" for one side's goals.
_HANDICAP = re.compile(r"^([12]) \(([+-]\d+(?:\.\d+)?)\)$")
_TEAM_GOALS = re.compile(r"^([12]) Over (\d+(?:\.\d+)?)$")


def _parametrised(market: str, selection: str, home_goals: int, away_goals: int) -> bool | None:
    """Settle the markets whose selection carries a line in its name."""
    if market == "handicap":
        match = _HANDICAP.match(selection)
        if match is None:
            return None
        side, line = match.group(1), float(match.group(2))
        if side == "1":
            return home_goals + line > away_goals
        return away_goals + line > home_goals

    if market == "team_goals":
        match = _TEAM_GOALS.match(selection)
        if match is None:
            return None
        side, line = match.group(1), float(match.group(2))
        return (home_goals if side == "1" else away_goals) > line

    return None


# Half-based and in-play markets - either half, first half, half time/full
# time, leading at any point - cannot be read off a final score, so they settle
# as void rather than being guessed at. Recording them as wins or losses on
# nothing but the full-time result would put unverified claims in the
# published record, which is the one thing this module exists to prevent.
def settle_selection(market: str, selection: str, home_goals: int, away_goals: int) -> str:
    total = home_goals + away_goals
    won = {
        ("home_win", "1"): home_goals > away_goals,
        ("away_win", "2"): away_goals > home_goals,
        ("draws", "X"): home_goals == away_goals,
        ("double_chance", "1X"): home_goals >= away_goals,
        ("double_chance", "X2"): away_goals >= home_goals,
        ("double_chance", "12"): home_goals != away_goals,
        ("btts", "GG"): home_goals > 0 and away_goals > 0,
        ("over_1_5", "Over 1.5"): total > 1.5,
        ("over_2_5", "Over 2.5"): total > 2.5,
        ("under_3_5", "Under 3.5"): total < 3.5,
        ("win_to_nil", "1 WTN"): home_goals > away_goals and away_goals == 0,
        ("win_to_nil", "2 WTN"): away_goals > home_goals and home_goals == 0,
        ("clean_sheet", "1 CS"): away_goals == 0,
        ("clean_sheet", "2 CS"): home_goals == 0,
        ("odd_even", "Odd"): total % 2 == 1,
        ("odd_even", "Even"): total % 2 == 0,
    }.get((market, selection))

    if won is None:
        won = _parametrised(market, selection, home_goals, away_goals)
    if won is None:
        return "void"
    return "won" if won else "lost"


def settle_pending_tips(db: Session) -> dict[str, int]:
    """Score every pending tip whose match has finished."""
    pending = db.scalars(
        select(PublishedTip)
        .join(Match, PublishedTip.match_id == Match.id)
        .where(
            PublishedTip.result == "pending",
            Match.status.in_(FINISHED_STATUSES),
            Match.home_goals.is_not(None),
            Match.away_goals.is_not(None),
        )
    ).all()

    counts = {"won": 0, "lost": 0, "void": 0}
    for tip in pending:
        outcome = settle_selection(
            tip.market, tip.selection, tip.match.home_goals, tip.match.away_goals
        )
        tip.result = outcome
        tip.settled_at = datetime.now(timezone.utc).replace(tzinfo=None)
        counts[outcome] += 1

    db.commit()
    return {**counts, "settled": len(pending)}


def performance(db: Session, days: int = 30, market: str | None = None) -> dict[str, Any]:
    """Strike rate and ROI over settled tips, at flat stakes.

    ROI is the honest headline: a strike rate can look strong while returns are
    negative, because short-priced winners do not cover the losers.
    """
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    stmt = select(PublishedTip).where(
        PublishedTip.published_at >= since,
        PublishedTip.result.in_(["won", "lost"]),
    )
    if market:
        stmt = stmt.where(PublishedTip.market == market)
    tips = db.scalars(stmt).all()

    if not tips:
        return {
            "settled": 0, "won": 0, "lost": 0, "strike_rate": None,
            "roi_percent": None, "profit_units": 0.0, "days": days,
            "note": "No settled tips in this window yet.",
        }

    won = [t for t in tips if t.result == "won"]
    returned = sum(t.odds for t in won)
    staked = float(len(tips))

    return {
        "settled": len(tips),
        "won": len(won),
        "lost": len(tips) - len(won),
        "strike_rate": round(len(won) / len(tips) * 100, 2),
        "roi_percent": round((returned - staked) / staked * 100, 2),
        "profit_units": round(returned - staked, 2),
        "average_odds": round(sum(t.odds for t in tips) / len(tips), 2),
        "days": days,
    }


def performance_by_market(db: Session, days: int = 30) -> list[dict[str, Any]]:
    markets = db.scalars(
        select(PublishedTip.market)
        .where(PublishedTip.result.in_(["won", "lost"]))
        .group_by(PublishedTip.market)
    ).all()
    return [{"market": m, **performance(db, days=days, market=m)} for m in markets]


def recent_settled(db: Session, limit: int = 20) -> list[PublishedTip]:
    return list(
        db.scalars(
            select(PublishedTip)
            .where(PublishedTip.result != "pending")
            .order_by(PublishedTip.kickoff_time.desc())
            .limit(limit)
        )
    )


def publish_tip(
    db: Session, match: Match, tip: dict[str, Any], is_vip: bool = False
) -> PublishedTip | None:
    """Record a tip before kickoff. Refuses to publish once a match has started.

    Without this guard the tracker could be back-filled with picks chosen after
    the result was known, which would make every published number meaningless.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if match.kickoff_time <= now:
        return None

    existing = db.scalar(
        select(PublishedTip).where(
            PublishedTip.match_id == match.id,
            PublishedTip.market == tip["market"],
            PublishedTip.selection == tip["selection"],
        )
    )
    if existing:
        return existing

    record = PublishedTip(
        match_id=match.id,
        market=tip["market"],
        selection=tip["selection"],
        probability=tip["probability"],
        odds=tip["odds"],
        rationale=tip.get("rationale"),
        is_vip=is_vip,
        kickoff_time=match.kickoff_time,
    )
    db.add(record)
    db.commit()
    return record
