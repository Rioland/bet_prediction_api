"""Daily betting slips.

The site assembles slips from the day's selections so something is always
published, and an admin may attach a real bookmaker booking code to any of
them. The two paths coexist deliberately: codes need a human to create the slip
on the bookmaker, and that human is not always around.

Nothing here generates a booking code. A code is issued by the bookmaker when a
slip is created on their platform - it is a key into their system, not a
computed value. A fabricated one resolves to nothing, or to an unrelated
stranger's slip, so a slip without a code publishes its selections instead and
says so.

Slips are persisted rather than rebuilt per request. A published code must keep
describing the selections it was created for.
"""

from __future__ import annotations

from datetime import date as date_type, datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import BettingSlip, Match

# How many slips to publish a day.
CARD_SIZE = 10

# Slip shapes, in the order they appear. Two kinds:
#
#   * leg-count and market-themed slips, which mean something regardless of
#     whether a bookmaker price is available
#   * target_odds slips, which name a price and are therefore only published
#     when every leg has a real market price behind it
#
# They are interleaved rather than grouped: the card is capped at ten, so
# price-target tiers listed last would never be reached once odds exist. A
# suppressed tier costs no slot, so the same order works with and without them.
TIERS: list[dict[str, Any]] = [
    {"tier": "banker", "legs": 1, "label": "Banker of the day"},
    {"tier": "double", "legs": 2, "label": "Two-fold"},
    {"tier": "2_odds", "legs": 2, "label": "2 odds", "target_odds": 2.0},
    {"tier": "treble", "legs": 3, "label": "Treble"},
    {"tier": "3_odds", "legs": 3, "label": "3 odds", "target_odds": 3.0},
    {"tier": "goals", "legs": 3, "label": "Goals special",
     "markets": {"over_1_5", "over_2_5", "under_3_5"}},
    {"tier": "safe_double_chance", "legs": 3, "label": "Double chance treble",
     "markets": {"double_chance"}},
    {"tier": "acca_4", "legs": 4, "label": "4-fold accumulator"},
    {"tier": "5_odds", "legs": 4, "label": "5 odds", "target_odds": 5.0},
    {"tier": "outright", "legs": 3, "label": "Match winners",
     "markets": {"home_win", "away_win"}},
    {"tier": "acca_5", "legs": 5, "label": "5-fold accumulator"},
    {"tier": "10_odds", "legs": 5, "label": "10 odds", "target_odds": 10.0},
    {"tier": "acca_6", "legs": 6, "label": "6-fold accumulator"},
    {"tier": "jackpot", "legs": 7, "label": "Long shot"},
    {"tier": "acca_8", "legs": 8, "label": "8-fold accumulator"},
]


def _leg_from(entry: dict[str, Any]) -> dict[str, Any]:
    """One selection, frozen as it was when the slip was built."""
    pick = entry["recommendation"]
    # With no market price the tip engine quotes fair odds - 1 / probability,
    # carrying no bookmaker margin. Those are an estimate of what a price
    # should be, not what anyone will actually offer.
    priced = pick.get("value", 0.0) != 0.0 or entry.get("priced_markets")
    return {
        "market_priced": bool(priced),
        "fixture_id": entry["fixture_id"],
        "home_team": entry["home_team"],
        "away_team": entry["away_team"],
        "league_name": entry.get("league_name"),
        "kickoff": entry["kickoff"],
        "market": pick["market"],
        "selection": pick["selection"],
        "label": pick["label"],
        "odds": pick["odds"],
        "probability": pick["probability"],
    }


def _assemble(entries: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any] | None:
    """Pick legs for one slip shape.

    Safest-first for plain accumulators. Where a price is targeted, legs are
    added until it is met, so "2 odds" is a genuine 2.0 rather than a label.
    """
    pool = entries
    wanted_markets = spec.get("markets")
    if wanted_markets:
        pool = [e for e in entries if e["recommendation"]["market"] in wanted_markets]

    ordered = sorted(pool, key=lambda e: e["recommendation"]["probability"], reverse=True)
    target = spec.get("target_odds")
    legs: list[dict[str, Any]] = []
    odds = 1.0
    probability = 1.0

    if target is None:
        chosen = ordered[: spec["legs"]]
        if len(chosen) < spec["legs"]:
            return None
        for entry in chosen:
            leg = _leg_from(entry)
            legs.append(leg)
            odds *= leg["odds"]
            probability *= leg["probability"]
    else:
        for entry in ordered:
            if odds >= target:
                break
            leg = _leg_from(entry)
            legs.append(leg)
            odds *= leg["odds"]
            probability *= leg["probability"]
        # A target the day's card cannot reach is not published as if it had.
        if odds < target or not legs:
            return None

    estimated = any(not leg["market_priced"] for leg in legs)
    # A tier whose whole point is reaching a price cannot be published from
    # estimated prices: "2 odds" would name a number no bookmaker offers.
    if target is not None and estimated:
        return None

    return {
        "tier": spec["tier"],
        "label": spec["label"],
        "legs": legs,
        "total_odds": round(odds, 2),
        "combined_probability": round(probability, 4),
        "odds_are_estimates": estimated,
    }


def build_slips(
    db: Session, entries: list[dict[str, Any]], sport: str, slip_date: date_type
) -> list[BettingSlip]:
    """Create the day's slips, leaving any that already exist untouched.

    Existing slips are never rebuilt: one may already carry a booking code, and
    the code has to keep describing the same legs.
    """
    if not entries:
        return []

    existing = {
        slip.tier: slip
        for slip in db.scalars(
            select(BettingSlip).where(
                BettingSlip.sport == sport, BettingSlip.slip_date == slip_date
            )
        )
    }

    published = len(existing)
    for spec in TIERS:
        if published >= CARD_SIZE:
            break
        if spec["tier"] in existing:
            continue
        assembled = _assemble(entries, spec)
        if assembled is None:
            continue
        published += 1
        db.add(
            BettingSlip(
                sport=sport,
                slip_date=slip_date,
                tier=assembled["tier"],
                label=assembled["label"],
                legs=assembled["legs"],
                total_odds=assembled["total_odds"],
                combined_probability=assembled["combined_probability"],
                odds_are_estimates=assembled["odds_are_estimates"],
            )
        )
    db.commit()
    return list_slips(db, sport, slip_date)


def list_slips(db: Session, sport: str, slip_date: date_type) -> list[BettingSlip]:
    order = {spec["tier"]: i for i, spec in enumerate(TIERS)}
    slips = list(
        db.scalars(
            select(BettingSlip).where(
                BettingSlip.sport == sport, BettingSlip.slip_date == slip_date
            )
        )
    )
    return sorted(slips, key=lambda s: order.get(s.tier, 99))


def attach_code(db: Session, slip: BettingSlip, code: str, added_by: str) -> BettingSlip:
    """Record a booking code an admin created on the bookmaker."""
    slip.booking_code = code.strip().upper()
    slip.code_added_at = datetime.now(timezone.utc).replace(tzinfo=None)
    slip.code_added_by = added_by
    db.commit()
    db.refresh(slip)
    return slip


def remove_code(db: Session, slip: BettingSlip) -> BettingSlip:
    slip.booking_code = None
    slip.code_added_at = None
    slip.code_added_by = None
    db.commit()
    db.refresh(slip)
    return slip


def settle_slips(db: Session, sport: str = "football") -> dict[str, int]:
    """Score slips whose legs have all finished.

    A slip wins only if every leg wins, so one lost leg settles it immediately
    even while other legs are unplayed.
    """
    from app.services.results import settle_selection

    pending = db.scalars(
        select(BettingSlip).where(BettingSlip.sport == sport, BettingSlip.result == "pending")
    ).all()

    counts = {"won": 0, "lost": 0, "settled": 0}
    for slip in pending:
        outcomes = []
        for leg in slip.legs:
            match = db.scalar(select(Match).where(Match.external_id == leg["fixture_id"]))
            if match is None or match.home_goals is None or match.away_goals is None:
                outcomes.append("pending")
                continue
            outcomes.append(
                settle_selection(leg["market"], leg["selection"], match.home_goals, match.away_goals)
            )

        if "lost" in outcomes:
            result = "lost"
        elif "pending" in outcomes:
            continue  # still running
        elif all(o == "won" for o in outcomes):
            result = "won"
        else:
            result = "void"

        slip.result = result
        slip.settled_at = datetime.now(timezone.utc).replace(tzinfo=None)
        counts[result] = counts.get(result, 0) + 1
        counts["settled"] += 1

    db.commit()
    return counts


def to_dict(slip: BettingSlip, can_see_code: bool = True) -> dict[str, Any]:
    """Serialise a slip.

    The booking code is what subscribers pay for, so it is only included for
    viewers with access. Everyone else still learns that a code exists - that
    is the reason to subscribe - but not what it is.
    """
    has_code = slip.booking_code is not None
    return {
        "id": slip.id,
        "tier": slip.tier,
        "label": slip.label,
        "sport": slip.sport,
        "date": slip.slip_date.isoformat(),
        "legs": slip.legs,
        "leg_count": len(slip.legs or []),
        "total_odds": slip.total_odds,
        # True when no bookmaker price backed some leg, so the total is a fair
        # estimate rather than a price anyone is offering.
        "odds_are_estimates": bool(slip.odds_are_estimates),
        "combined_probability": slip.combined_probability,
        "booking_code": slip.booking_code if can_see_code else None,
        # False means "build it yourself from the selections", never a fake code.
        "has_code": has_code,
        "code_locked": has_code and not can_see_code,
        "result": slip.result,
    }
