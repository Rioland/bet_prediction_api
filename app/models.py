from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    LargeBinary,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import text
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="user")        # user | moderator | admin | super_admin
    status = Column(String(50), default="active")    # active | suspended | banned
    subscription_type = Column(String(50), default="free")  # free | premium
    two_factor_enabled = Column(Boolean, default=False)
    avatar = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)


# ── Fixture history ───────────────────────────────────────────────────────────
# Predictions from the live ESPN/football-data feed need no persistence, but a
# model that learns from results does: these tables are what the training
# pipeline reads.


# Every competition and match belongs to exactly one sport. Defaulting to
# football keeps rows written before this column existed valid.
DEFAULT_SPORT = "football"


class League(Base):
    __tablename__ = "leagues"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(Integer, unique=True, index=True, nullable=True)
    name = Column(String(120), nullable=False)
    country = Column(String(120), nullable=True)
    sport = Column(String(20), index=True, nullable=False, default=DEFAULT_SPORT,
                   server_default=DEFAULT_SPORT)


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(Integer, unique=True, index=True, nullable=True)
    name = Column(String(120), index=True, nullable=False)
    logo_url = Column(String(512), nullable=True)


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(Integer, unique=True, index=True, nullable=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), index=True, nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), index=True, nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), index=True, nullable=False)
    kickoff_time = Column(DateTime, index=True, nullable=False)
    status = Column(String(50), index=True, nullable=False)
    season = Column(Integer, index=True, nullable=True)
    sport = Column(String(20), index=True, nullable=False, default=DEFAULT_SPORT,
                   server_default=DEFAULT_SPORT)

    # Final result; NULL until played. These are the training labels. For
    # basketball these hold points rather than goals - the column names are kept
    # so the shared loaders and the existing rows do not need rewriting.
    home_goals = Column(Integer, nullable=True)
    away_goals = Column(Integer, nullable=True)

    # Post-match team stats. Used only to build rolling averages for *later*
    # fixtures - never as features for the match they belong to.
    home_shots_on_target = Column(Integer, nullable=True)
    away_shots_on_target = Column(Integer, nullable=True)
    home_possession = Column(Float, nullable=True)
    away_possession = Column(Float, nullable=True)
    home_corners = Column(Integer, nullable=True)
    away_corners = Column(Integer, nullable=True)

    # Closing 1X2 odds, for value detection and as the benchmark to beat.
    odds_home = Column(Float, nullable=True)
    odds_draw = Column(Float, nullable=True)
    odds_away = Column(Float, nullable=True)

    league = relationship("League")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])


class PublishedTip(Base):
    """A tip as published, recorded before kickoff.

    The results page is only worth anything if picks cannot be chosen or edited
    once the result is known, so nothing here is rewritten after settlement
    except the outcome fields.
    """

    __tablename__ = "published_tips"
    __table_args__ = (UniqueConstraint("match_id", "market", "selection", name="uq_tip_selection"),)

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), index=True, nullable=False)
    sport = Column(String(20), index=True, nullable=False, default=DEFAULT_SPORT,
                   server_default=DEFAULT_SPORT)
    market = Column(String(40), index=True, nullable=False)
    selection = Column(String(40), nullable=False)
    probability = Column(Float, nullable=False)
    odds = Column(Float, nullable=False)
    is_vip = Column(Boolean, default=False, index=True)
    rationale = Column(String(400), nullable=True)
    published_at = Column(DateTime, default=datetime.utcnow, index=True)
    kickoff_time = Column(DateTime, index=True, nullable=False)
    result = Column(String(20), default="pending", index=True)  # pending|won|lost|void
    settled_at = Column(DateTime, nullable=True)

    match = relationship("Match")


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String(200), unique=True, index=True, nullable=False)
    title = Column(String(250), nullable=False)
    excerpt = Column(String(500), nullable=True)
    body = Column(Text, nullable=False)
    cover_image = Column(String(512), nullable=True)
    author = Column(String(120), nullable=True)
    published = Column(Boolean, default=False, index=True)
    published_at = Column(DateTime, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BettingSlip(Base):
    """A multi-leg slip built from the day's picks.

    Slips are persisted rather than rebuilt per request. An admin may attach a
    real SportyBet booking code to one, and that code has to keep pointing at
    the selections it was created for - regenerating on the fly would leave a
    published code describing a different slip.

    booking_code is NULL until an admin supplies a real code. Nothing generates
    one: a booking code is issued by the bookmaker when a slip is created on
    their platform, so a fabricated value would resolve to nothing or to an
    unrelated slip.
    """

    __tablename__ = "betting_slips"
    __table_args__ = (UniqueConstraint("sport", "slip_date", "tier", name="uq_slip_tier"),)

    id = Column(Integer, primary_key=True, index=True)
    sport = Column(String(20), index=True, nullable=False, default=DEFAULT_SPORT,
                   server_default=DEFAULT_SPORT)
    slip_date = Column(Date, index=True, nullable=False)
    # "banker" | "2_odds" | "acca_5" ... - the shape of the slip, one per day.
    tier = Column(String(40), nullable=False)
    label = Column(String(120), nullable=False)
    legs = Column(JSON, nullable=False, default=list)
    total_odds = Column(Float, nullable=False)
    # True when some leg had no bookmaker price, so total_odds is a fair
    # estimate (1 / probability, no margin) rather than an offered price.
    odds_are_estimates = Column(Boolean, nullable=False, default=False,
                                server_default=text("false"))
    combined_probability = Column(Float, nullable=False)

    # Supplied by an admin who created the slip on the bookmaker. Never generated.
    booking_code = Column(String(40), nullable=True)
    code_added_at = Column(DateTime, nullable=True)
    code_added_by = Column(String(255), nullable=True)

    result = Column(String(20), default="pending", index=True)  # pending|won|lost|void
    settled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Subscriptions and payments ───────────────────────────────────────────────


class AppSetting(Base):
    """Admin-editable configuration that must survive a restart.

    Settings previously lived in a module-level dict, so an admin changing the
    subscription price would see it silently revert the next time the server
    restarted.
    """

    __tablename__ = "app_settings"

    key = Column(String(80), primary_key=True)
    value = Column(String(500), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(255), nullable=True)


class Payment(Base):
    """One checkout attempt with the payment gateway.

    The amount is fixed when checkout starts. If an admin changes the price
    while a customer is mid-payment, the customer is charged, and verified
    against, the price they were shown - not the new one.
    """

    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    provider = Column(String(30), nullable=False, default="opay")
    # Our order reference, sent to the gateway. Random, so it cannot be guessed.
    reference = Column(String(64), unique=True, index=True, nullable=False)
    provider_order_no = Column(String(64), nullable=True, index=True)
    # Kobo. The gateway works in the currency's minor unit.
    amount_kobo = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="NGN")
    status = Column(String(20), nullable=False, default="initial", index=True)
    checkout_url = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    paid_at = Column(DateTime, nullable=True)
    # Set once the subscription has been extended for this payment, so a
    # retried callback cannot extend it twice.
    applied_at = Column(DateTime, nullable=True)
    failure_reason = Column(String(500), nullable=True)

    user = relationship("User")


class Subscription(Base):
    """A user's paid access window.

    Access is decided by expires_at alone. Nothing marks a subscription active
    that is not backed by a verified payment.
    """

    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, index=True, nullable=False)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False, index=True)
    last_payment_id = Column(Integer, ForeignKey("payments.id"), nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")


class ModelArtifact(Base):
    """A trained model file, kept in the database as well as on disk.

    Hosts with ephemeral filesystems (Render's free tier, most containers)
    delete ./models on every restart. The database copy lets a fresh instance
    restore the last trained models instead of serving 503 until someone
    retrains.
    """

    __tablename__ = "model_artifacts"

    filename = Column(String(120), primary_key=True)
    data = Column(LargeBinary, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
