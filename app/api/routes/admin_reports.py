from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select

from app.api.deps import DbSession
from app.models.entities import Report, ReportStatus, User, UserRole
from app.schemas.admin import ReportOut, ResolveReportRequest
from app.services.audit import log_admin_action
from app.services.rbac import require_role

router = APIRouter(prefix="/admin/reports", tags=["admin-reports"])


@router.get("", response_model=list[ReportOut])
def get_reports(db: DbSession, _: Annotated[User, Depends(require_role(UserRole.MODERATOR))]) -> list[Report]:
    return list(db.scalars(select(Report).order_by(Report.created_at.desc())))


@router.patch("/{report_id}/resolve")
def resolve_report(
    report_id: int,
    payload: ResolveReportRequest,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.MODERATOR))],
) -> dict:
    report = db.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    report.status = ReportStatus.RESOLVED
    report.moderation_notes = payload.moderation_notes
    report.resolved_by = current_admin.id
    report.resolved_at = datetime.utcnow()
    db.commit()
    log_admin_action(
        db,
        current_admin,
        "resolve_report",
        target_user_id=report.target_user_id,
        ip_address=request.client.host if request.client else None,
        metadata={"report_id": report_id},
    )
    return {"status": "resolved"}
