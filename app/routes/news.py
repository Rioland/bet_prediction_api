"""Article feed. Public reads; writes require an admin."""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Article, User
from app.routes.admin_auth import get_current_admin

router = APIRouter(prefix="/news", tags=["news"])
DbSession = Annotated[Session, Depends(get_db)]


class ArticleSummary(BaseModel):
    id: int
    slug: str
    title: str
    excerpt: str | None = None
    cover_image: str | None = None
    author: str | None = None
    published_at: datetime | None = None

    model_config = {"from_attributes": True}


class ArticleDetail(ArticleSummary):
    body: str


class ArticleInput(BaseModel):
    slug: str
    title: str
    body: str
    excerpt: str | None = None
    cover_image: str | None = None
    author: str | None = None
    published: bool = False


@router.get("", response_model=list[ArticleSummary])
def list_articles(
    db: DbSession,
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
):
    return list(
        db.scalars(
            select(Article)
            .where(Article.published.is_(True))
            .order_by(Article.published_at.desc())
            .limit(limit)
            .offset(offset)
        )
    )


@router.get("/{slug}", response_model=ArticleDetail)
def get_article(slug: str, db: DbSession):
    article = db.scalar(select(Article).where(Article.slug == slug, Article.published.is_(True)))
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@router.post("", response_model=ArticleDetail, status_code=201)
def create_article(
    payload: ArticleInput, db: DbSession, _admin: Annotated[User, Depends(get_current_admin)]
):
    if db.scalar(select(Article).where(Article.slug == payload.slug)):
        raise HTTPException(status_code=400, detail="Slug already in use")

    article = Article(**payload.model_dump())
    if payload.published:
        article.published_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.add(article)
    db.commit()
    db.refresh(article)
    return article


@router.patch("/{slug}", response_model=ArticleDetail)
def update_article(
    slug: str,
    payload: ArticleInput,
    db: DbSession,
    _admin: Annotated[User, Depends(get_current_admin)],
):
    article = db.scalar(select(Article).where(Article.slug == slug))
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")

    was_published = article.published
    for field, value in payload.model_dump().items():
        setattr(article, field, value)
    # Stamp the publication date the first time it goes live, and only then.
    if payload.published and not was_published:
        article.published_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()
    db.refresh(article)
    return article
