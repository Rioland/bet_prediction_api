from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class MatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    league_id: int
    home_team_id: int
    away_team_id: int
    home_team_name: str | None = None
    away_team_name: str | None = None
    league_name: str | None = None
    kickoff_time: datetime
    status: str


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    match_id: int
    prediction_type: str
    prediction: str
    confidence: float
    probabilities: dict


class DeviceRegisterRequest(BaseModel):
    token: str
    platform: str
