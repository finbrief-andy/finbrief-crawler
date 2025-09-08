#!/usr/bin/env python3
"""
FinBrief API Server
Combines Authentication and Feedback APIs
"""

import os
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from jose import jwt
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc, and_
import uvicorn

# Import existing DB models and init function
from src.database.models_migration import (
    init_db_and_create,
    News,
    Analysis,
    Feedback,
    User,
    Strategy,
    ActionEnum,
    FeedbackTypeEnum,
    VoteEnum,
    StrategyHorizonEnum,
    MarketEnum,
    AssetTypeEnum,
)

# Init DB engine (reads DATABASE_URI env var or default)
DATABASE_URI = os.getenv("DATABASE_URI")
engine = init_db_and_create(DATABASE_URI)
SessionLocal = sessionmaker(bind=engine)

app = FastAPI(title="FinBrief API", description="Authentication and Feedback API")

# Auth configuration
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# -----------------
# Pydantic Schemas
# -----------------
class UserCreate(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

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
        from_attributes = True

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
        from_attributes = True

class StrategyResponse(BaseModel):
    id: int
    horizon: str
    market: str
    asset_focus: Optional[str]
    strategy_date: datetime
    title: str
    summary: str
    key_drivers: List[str]
    action_recommendations: List[Dict[str, Any]]
    confidence_score: Optional[float]
    generated_by: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

# -----------------
# Helpers
# -----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def hash_password(password: str):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = int(user_id_str)
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Could not validate token")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid user ID in token")
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

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
# Auth Endpoints
# -----------------
@app.post("/auth/signup", response_model=UserResponse)
def signup(user: UserCreate, db=Depends(get_db)):
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    u = User(email=user.email, password_hash=hash_password(user.password))
    db.add(u)
    db.commit()
    db.refresh(u)
    return u

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# -----------------
# Feedback Endpoints
# -----------------
@app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_feedback(payload: SaveFeedbackRequest, db=Depends(get_db), current_user: User = Depends(get_current_user)):
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

@app.post("/feedback/{feedback_id}/mark_processed")
def mark_feedback_processed(feedback_id: int, db=Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")
    fb.processed = True
    db.add(fb)
    db.commit()
    return {"status": "ok", "feedback_id": feedback_id}

# -----------------
# Strategy Endpoints
# -----------------
@app.get("/strategy/{horizon}", response_model=StrategyResponse)
def get_latest_strategy(horizon: str, market: str = "global", db=Depends(get_db)):
    """Get the latest strategy for a given horizon and market"""
    try:
        horizon_enum = StrategyHorizonEnum[horizon.lower()]
        market_enum = MarketEnum[market.lower()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid horizon or market")
    
    strategy = db.query(Strategy).filter(
        and_(
            Strategy.horizon == horizon_enum,
            Strategy.market == market_enum
        )
    ).order_by(desc(Strategy.strategy_date)).first()
    
    if not strategy:
        raise HTTPException(status_code=404, detail=f"No {horizon} strategy found for {market} market")
    
    return strategy

@app.get("/strategies", response_model=List[StrategyResponse])
def list_strategies(market: str = "global", limit: int = 10, db=Depends(get_db)):
    """List recent strategies for a market, grouped by horizon"""
    try:
        market_enum = MarketEnum[market.lower()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid market")
    
    strategies = db.query(Strategy).filter(
        Strategy.market == market_enum
    ).order_by(desc(Strategy.strategy_date)).limit(limit).all()
    
    return strategies

@app.post("/strategy/generate", response_model=Dict[str, Any])
def generate_new_strategies(market: str = "global", 
                          horizons: List[str] = None, 
                          current_user: User = Depends(get_current_user),
                          db=Depends(get_db)):
    """Generate new strategies for specified horizons and market (admin only)"""
    if current_user.role.value not in ["admin", "system"]:
        raise HTTPException(status_code=403, detail="Only admins can generate strategies")
    
    try:
        market_enum = MarketEnum[market.lower()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid market")
    
    from src.services.strategy_generator import StrategyGenerator
    generator = StrategyGenerator()
    
    results = {}
    target_horizons = horizons or ["daily", "weekly", "monthly", "yearly"]
    
    for horizon_str in target_horizons:
        try:
            horizon_enum = StrategyHorizonEnum[horizon_str.lower()]
            strategy = generator.create_strategy(db, horizon_enum, market_enum)
            if strategy:
                results[horizon_str] = {
                    "id": strategy.id,
                    "title": strategy.title,
                    "created_at": strategy.created_at.isoformat()
                }
        except KeyError:
            results[horizon_str] = {"error": "Invalid horizon"}
        except Exception as e:
            results[horizon_str] = {"error": str(e)}
    
    return {
        "market": market,
        "generated_strategies": results,
        "total_generated": len([r for r in results.values() if "id" in r])
    }

@app.get("/strategy/{horizon}/history", response_model=List[StrategyResponse])
def get_strategy_history(horizon: str, market: str = "global", limit: int = 7, db=Depends(get_db)):
    """Get historical strategies for a given horizon"""
    try:
        horizon_enum = StrategyHorizonEnum[horizon.lower()]
        market_enum = MarketEnum[market.lower()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid horizon or market")
    
    strategies = db.query(Strategy).filter(
        and_(
            Strategy.horizon == horizon_enum,
            Strategy.market == market_enum
        )
    ).order_by(desc(Strategy.strategy_date)).limit(limit).all()
    
    return strategies

# -----------------
# Root endpoint
# -----------------
@app.get("/")
def read_root():
    return {
        "message": "FinBrief API", 
        "version": "1.0.0",
        "endpoints": {
            "auth": ["/auth/signup", "/auth/login", "/auth/me"],
            "feedback": ["/feedback", "/analysis/{analysis_id}/feedback", "/news/{news_id}/analysis", "/feedback/{feedback_id}"],
            "strategy": ["/strategy/{horizon}", "/strategies", "/strategy/generate", "/strategy/{horizon}/history"],
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)