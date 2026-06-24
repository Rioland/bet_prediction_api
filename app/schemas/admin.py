from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr

from app.models.entities import ReportStatus, SubscriptionType, UserRole, UserStatus


class AdminLoginRequest(BaseModel):
    email: EmailStr
    password: str
    otp_code: str | None = None


class AdminUserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: EmailStr
    role: UserRole
    status: UserStatus
    subscription_type: SubscriptionType
    created_at: datetime


class AdminSessionResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    csrf_token: str
    user: AdminUserOut


class AdminUserPatch(BaseModel):
    name: str | None = None
    role: UserRole | None = None
    status: UserStatus | None = None


class UserActionRequest(BaseModel):
    reason: str | None = None


class NotificationSendRequest(BaseModel):
    title: str
    body: str
    channel: str = "in_app"
    audience: str = "all"
    user_ids: list[int] = []


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    reporter_user_id: int
    target_user_id: int | None
    category: str
    message: str
    status: ReportStatus
    moderation_notes: str | None
    created_at: datetime


class ResolveReportRequest(BaseModel):
    moderation_notes: str


class SettingPatchRequest(BaseModel):
    key: str
    value: str


class TwoFASetupResponse(BaseModel):
    secret: str
    otpauth_url: str


class TwoFAVerifyRequest(BaseModel):
    otp_code: str
