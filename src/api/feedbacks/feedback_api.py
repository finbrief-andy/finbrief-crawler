"""
FinBrief Feedback API (FastAPI)

Purpose:
- Provide endpoints to capture user feedback on generated analyses (action strategies).
- Minimal-auth approach (no JWT) for MVP: user_id optional. Later we can add authentication.

Pre-reqs:
- models_migration.py (from earlier) must be importable in the same folder or installed as module.
- DB created via models_migration.init_db_and_create

Run:
- pip install fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv
- export DATABASE_URI="postgresql://postgres:postgres@localhost:5432/finance_app"
- python models_migration.py   # if not already run
- uvicorn feedback_api:app --reload --port 8000

Endpoints:
- POST /feedback                -> submit feedback for an analysis
- GET  /analysis/{analysis_id}/feedback -> list feedback for an analysis
- GET  /news/{news_id}/analysis -> list analyses for a news item
- GET  /feedback/{feedback_id}  -> fetch a single feedback entry

Notes:
- This is intentionally minimal to get feedback data flowing into DB fast.
- Later additions: authentication (JWT), rate limiting, spam protection, validation by user role, moderation UI.
"""

from typing import Optional, List
from datetime import datetime
import os

from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import sessionmaker
from ..auth.auth_api import get_current_user

# Import existing DB models and init function
from src.database.models_migration import (
    init_db_and_create,
    News,
    Analysis,
    Feedback,
    User,
    ActionEnum,
    FeedbackTypeEnum,
    VoteEnum,
)

# Init DB engine (reads DATABASE_URI env var or default)
DATABASE_URI = os.getenv("DATABASE_URI")
engine = init_db_and_create(DATABASE_URI)
SessionLocal = sessionmaker(bind=engine)

app = FastAPI(title="FinBrief Feedback API")

# -----------------
# Pydantic schemas
# -----------------
class SaveFeedbackRequest(BaseModel):
    analysis_id: int
    user_id: Optional[int] = None
    feedback_type: Optional[str] = Field(default="vote", description="vote|rating|comment")
    vote: Optional[str] = None  # agree|disagree|neutral
    rating: Optional[int] = None  # 1..5
    selected_action_short: Optional[str] = None
    selected_action_mid: Optional[str] = None
    selected_action_long: Optional[str] = None
    comment: Optional[str] = None

    @validator("feedback_type")
    def check_feedback_type(cls, v):
        if v is None:
            return v
        if v.lower() not in {"vote", "rating", "comment"}:
            raise ValueError("feedback_type must be one of vote|rating|comment")
        return v.lower()

    @validator("rating")
    def check_rating(cls, v):
        if v is None:
            return v
        if not (1 <= v <= 5):
            raise ValueError("rating must be between 1 and 5")
        return v

class FeedbackResponse(BaseModel):
    id: int
    analysis_id: int
    user_id: Optional[int]
    feedback_type: str
    vote: Optional[str]
    rating: Optional[int]
    selected_action_short: Optional[str]
    selected_action_mid: Optional[str]
    selected_action_long: Optional[str]
    comment: Optional[str]
    created_at: datetime
    processed: bool

    class Config:
        orm_mode = True

class AnalysisResponse(BaseModel):
    id: int
    news_id: int
    sentiment: Optional[str]
    action_short: Optional[str]
    action_mid: Optional[str]
    action_long: Optional[str]
    model_version: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True

# -----------------
# Helpers
# -----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def map_to_action_enum(val: Optional[str]):
    if val is None:
        return None
    key = val.strip().upper()
    try:
        return ActionEnum[key]
    except Exception:
        return None


def map_to_vote_enum(val: Optional[str]):
    if val is None:
        return None
    key = val.strip().lower()
    mapping = {"agree": VoteEnum.agree, "disagree": VoteEnum.disagree, "neutral": VoteEnum.neutral}
    return mapping.get(key)


def map_to_feedback_type(val: Optional[str]):
    if val is None:
        return FeedbackTypeEnum.vote
    key = val.strip().lower()
    mapping = {"vote": FeedbackTypeEnum.vote, "rating": FeedbackTypeEnum.rating, "comment": FeedbackTypeEnum.comment}
    return mapping.get(key, FeedbackTypeEnum.vote)

# -----------------
# Endpoints
# -----------------
@app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_feedback(payload: SaveFeedbackRequest, db=Depends(get_db), current_user: User = Depends(get_current_user)):
    # Dùng current_user.id thay cho payload.user_id
    fb = Feedback(
        analysis_id=payload.analysis_id,
        user_id=current_user.id,
        feedback_type=map_to_feedback_type(payload.feedback_type),
        vote=map_to_vote_enum(payload.vote),
        rating=payload.rating,
        selected_action_short=map_to_action_enum(payload.selected_action_short),
        selected_action_mid=map_to_action_enum(payload.selected_action_mid),
        selected_action_long=map_to_action_enum(payload.selected_action_long),
        comment=payload.comment,
        processed=False,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb


@app.get("/analysis/{analysis_id}/feedback", response_model=List[FeedbackResponse])
def list_feedback_for_analysis(analysis_id: int, db=Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    rows = db.query(Feedback).filter(Feedback.analysis_id == analysis_id).order_by(Feedback.created_at.desc()).all()
    return rows


@app.get("/news/{news_id}/analysis", response_model=List[AnalysisResponse])
def list_analyses_for_news(news_id: int, db=Depends(get_db)):
    news = db.query(News).filter(News.id == news_id).first()
    if not news:
        raise HTTPException(status_code=404, detail="News not found")
    rows = db.query(Analysis).filter(Analysis.news_id == news_id).order_by(Analysis.created_at.desc()).all()
    return rows


@app.get("/feedback/{feedback_id}", response_model=FeedbackResponse)
def get_feedback(feedback_id: int, db=Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return fb

# -----------------
# Optional: endpoint to mark feedback as processed by training pipeline
# -----------------
@app.post("/feedback/{feedback_id}/mark_processed")
def mark_feedback_processed(feedback_id: int, db=Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")
    fb.processed = True
    db.add(fb)
    db.commit()
    return {"status": "ok", "feedback_id": feedback_id}
