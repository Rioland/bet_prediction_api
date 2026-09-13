"""Betting slips and booking codes.

The load-bearing rule: a booking code is only ever a real one recorded by an
admin. Nothing generates one, because a code is a key into the bookmaker's
system and a fabricated value resolves to nothing or to a stranger's slip.
"""

from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.auth import create_access_token, hash_password
from app.database import get_db
from app.models import BettingSlip, League, Match, Team, User
from app.services import slips as slips_service
from main import app


@pytest.fixture()
def client(db_session: Session) -> TestClient:
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _admin(db: Session) -> User:
    user = User(name="A", email="slipadmin@test.com",
                password_hash=hash_password("password1234"), role="admin")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _bearer(user: User) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def _entry(fixture_id: int, market: str, selection: str, odds: float, probability: float,
           priced: bool = True) -> dict:
    return {
        "fixture_id": fixture_id,
        "home_team": f"H{fixture_id}",
        "away_team": f"A{fixture_id}",
        "league_name": "League",
        "kickoff": "2026-09-12T15:00:00",
        "recommendation": {
            "market": market, "selection": selection, "label": f"{selection} pick",
            "odds": odds, "probability": probability,
            "value": 0.05 if priced else 0.0,
        },
    }


def _pool(priced: bool = True) -> list[dict]:
    return [
        _entry(i, "over_1_5" if i % 2 else "double_chance", "Over 1.5" if i % 2 else "1X",
               1.5 + i * 0.1, 0.9 - i * 0.02, priced)
        for i in range(12)
    ]


# --- building ----------------------------------------------------------------


def test_slips_are_built_from_the_pool(db_session: Session) -> None:
    built = slips_service.build_slips(db_session, _pool(), "football", date(2026, 9, 12))
    assert len(built) == slips_service.CARD_SIZE
    assert all(s.legs for s in built)


def test_no_slip_is_created_with_a_booking_code(db_session: Session) -> None:
    """Codes come from a human on the bookmaker, never from the builder."""
    built = slips_service.build_slips(db_session, _pool(), "football", date(2026, 9, 12))
    assert all(s.booking_code is None for s in built)


def test_rebuilding_leaves_existing_slips_untouched(db_session: Session) -> None:
    """A published code must keep describing the legs it was created for."""
    slip_date = date(2026, 9, 12)
    first = slips_service.build_slips(db_session, _pool(), "football", slip_date)
    slips_service.attach_code(db_session, first[0], "SPB123", "admin@test.com")
    original_legs = list(first[0].legs)

    slips_service.build_slips(db_session, _pool()[::-1], "football", slip_date)
    db_session.refresh(first[0])

    assert first[0].booking_code == "SPB123"
    assert first[0].legs == original_legs


def test_targeted_odds_tiers_are_withheld_without_real_prices(db_session: Session) -> None:
    """"2 odds" must not name a price derived from fair odds with no margin."""
    built = slips_service.build_slips(db_session, _pool(priced=False), "football", date(2026, 9, 12))
    tiers = {s.tier for s in built}
    assert not tiers & {"2_odds", "3_odds", "5_odds", "10_odds"}
    assert all(s.odds_are_estimates for s in built)


def test_priced_pools_do_produce_targeted_tiers(db_session: Session) -> None:
    built = slips_service.build_slips(db_session, _pool(priced=True), "football", date(2026, 9, 12))
    assert any(s.tier.endswith("_odds") for s in built)
    assert all(not s.odds_are_estimates for s in built)


def test_market_themed_slips_only_use_that_market(db_session: Session) -> None:
    built = slips_service.build_slips(db_session, _pool(), "football", date(2026, 9, 12))
    goals = next((s for s in built if s.tier == "goals"), None)
    assert goals is not None
    assert all(leg["market"] in {"over_1_5", "over_2_5", "under_3_5"} for leg in goals.legs)


def test_combined_probability_is_the_product_of_legs(db_session: Session) -> None:
    built = slips_service.build_slips(db_session, _pool(), "football", date(2026, 9, 12))
    treble = next(s for s in built if s.tier == "treble")
    expected = 1.0
    for leg in treble.legs:
        expected *= leg["probability"]
    assert treble.combined_probability == pytest.approx(expected, abs=1e-4)


def test_an_empty_pool_builds_nothing(db_session: Session) -> None:
    assert slips_service.build_slips(db_session, [], "football", date(2026, 9, 12)) == []


# --- codes -------------------------------------------------------------------


def test_public_slips_never_expose_a_generated_code(client: TestClient, db_session) -> None:
    slips_service.build_slips(db_session, _pool(), "football", datetime.utcnow().date())
    body = client.get("/slips").json()

    assert body["with_codes"] == 0
    for slip in body["slips"]:
        assert slip["booking_code"] is None
        assert slip["has_code"] is False


def test_admin_can_attach_and_clear_a_code(client: TestClient, db_session: Session) -> None:
    built = slips_service.build_slips(db_session, _pool(), "football", datetime.utcnow().date())
    headers = _bearer(_admin(db_session))

    attached = client.post(f"/admin/slips/{built[0].id}/code",
                           json={"booking_code": "spb4x9k2"}, headers=headers).json()
    assert attached["booking_code"] == "SPB4X9K2", "codes are normalised to upper case"
    assert client.get("/slips").json()["with_codes"] == 1

    client.delete(f"/admin/slips/{built[0].id}/code", headers=headers)
    assert client.get("/slips").json()["with_codes"] == 0


def test_attaching_a_code_requires_admin(client: TestClient, db_session: Session) -> None:
    built = slips_service.build_slips(db_session, _pool(), "football", datetime.utcnow().date())
    assert client.post(f"/admin/slips/{built[0].id}/code",
                       json={"booking_code": "SPB123"}).status_code == 401

    plain = User(name="U", email="plain@slip.com",
                 password_hash=hash_password("password1234"), role="user")
    db_session.add(plain)
    db_session.commit()
    db_session.refresh(plain)
    assert client.post(f"/admin/slips/{built[0].id}/code", json={"booking_code": "SPB123"},
                       headers=_bearer(plain)).status_code == 403


def test_malformed_codes_are_rejected(client: TestClient, db_session: Session) -> None:
    built = slips_service.build_slips(db_session, _pool(), "football", datetime.utcnow().date())
    headers = _bearer(_admin(db_session))
    for bad in ["ab", "has spaces", "!!!!", "x" * 50]:
        assert client.post(f"/admin/slips/{built[0].id}/code",
                           json={"booking_code": bad}, headers=headers).status_code == 422


def test_unknown_slip_is_404(client: TestClient, db_session: Session) -> None:
    assert client.post("/admin/slips/9999/code", json={"booking_code": "SPB123"},
                       headers=_bearer(_admin(db_session))).status_code == 404


# --- settlement --------------------------------------------------------------


def _played(db: Session, fixture_id: int, home: int, away: int) -> None:
    league = db.query(League).first() or League(external_id=1, name="L", country="C")
    db.add(league)
    db.flush()
    h, a = Team(name=f"H{fixture_id}"), Team(name=f"A{fixture_id}")
    db.add_all([h, a])
    db.flush()
    db.add(Match(external_id=fixture_id, league_id=league.id, home_team_id=h.id,
                 away_team_id=a.id, kickoff_time=datetime.utcnow() - timedelta(hours=3),
                 status="FT", season=2026, home_goals=home, away_goals=away))
    db.commit()


def test_one_losing_leg_settles_the_whole_slip(db_session: Session) -> None:
    slip = BettingSlip(
        sport="football", slip_date=date.today(), tier="double", label="Two-fold",
        legs=[{"fixture_id": 501, "market": "over_1_5", "selection": "Over 1.5"},
              {"fixture_id": 502, "market": "over_1_5", "selection": "Over 1.5"}],
        total_odds=2.0, combined_probability=0.5,
    )
    db_session.add(slip)
    db_session.commit()

    _played(db_session, 501, 0, 0)  # under 1.5 - this leg loses
    # 502 has not been played at all; the slip is already lost regardless.
    counts = slips_service.settle_slips(db_session)

    db_session.refresh(slip)
    assert slip.result == "lost"
    assert counts["lost"] == 1


def test_a_slip_with_unplayed_legs_stays_pending(db_session: Session) -> None:
    slip = BettingSlip(
        sport="football", slip_date=date.today(), tier="double", label="Two-fold",
        legs=[{"fixture_id": 601, "market": "over_1_5", "selection": "Over 1.5"},
              {"fixture_id": 602, "market": "over_1_5", "selection": "Over 1.5"}],
        total_odds=2.0, combined_probability=0.5,
    )
    db_session.add(slip)
    db_session.commit()

    _played(db_session, 601, 2, 1)  # wins; the other leg is unplayed
    slips_service.settle_slips(db_session)

    db_session.refresh(slip)
    assert slip.result == "pending"


def test_all_legs_winning_settles_the_slip_won(db_session: Session) -> None:
    slip = BettingSlip(
        sport="football", slip_date=date.today(), tier="double", label="Two-fold",
        legs=[{"fixture_id": 701, "market": "over_1_5", "selection": "Over 1.5"},
              {"fixture_id": 702, "market": "over_1_5", "selection": "Over 1.5"}],
        total_odds=2.0, combined_probability=0.5,
    )
    db_session.add(slip)
    db_session.commit()

    _played(db_session, 701, 2, 1)
    _played(db_session, 702, 3, 0)
    slips_service.settle_slips(db_session)

    db_session.refresh(slip)
    assert slip.result == "won"
