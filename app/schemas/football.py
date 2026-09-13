"""Response models matching lib/api-spec/openapi.yaml, so the generated
predictor-web client works against this API without regeneration."""

from pydantic import BaseModel


class LeagueOut(BaseModel):
    id: int
    name: str
    country: str | None = None
    logo: str | None = None
    season: int | None = None


class PredictionOut(BaseModel):
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    btts_prob: float
    over_25_prob: float
    predicted_winner: str
    # 0-1, matching the frontend helpers (formatPercent, getConfidenceColor).
    confidence: float
    home_xg: float
    away_xg: float
    predicted_score: str


class MatchWithPrediction(BaseModel):
    fixture_id: int
    league_id: int
    league_name: str
    league_logo: str | None = None
    league_country: str | None = None
    home_team: str
    home_logo: str | None = None
    away_team: str
    away_logo: str | None = None
    kickoff: str
    status: str
    home_score: int | None = None
    away_score: int | None = None
    prediction: PredictionOut
