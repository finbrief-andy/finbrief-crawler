# models_migration.py
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Boolean, Float,
    Enum as SAEnum, ForeignKey, func, UniqueConstraint, text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
import enum
import os

Base = declarative_base()

def get_json_type(engine_url=None):
    """Get appropriate JSON type based on database"""
    if engine_url and "postgresql" in engine_url:
        from sqlalchemy.dialects.postgresql import JSONB
        return JSONB
    return JSON

def get_array_type(engine_url=None):
    """Get appropriate array type based on database"""
    if engine_url and "postgresql" in engine_url:
        from sqlalchemy.dialects.postgresql import ARRAY
        return ARRAY(Text)
    return Text  # Store as JSON string for SQLite

# ------------------------
# Enums
# ------------------------
class RoleEnum(enum.Enum):
    user = "user"
    admin = "admin"
    analyst = "analyst"
    system = "system"

class SentimentEnum(enum.Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"

class ActionEnum(enum.Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    AVOID = "AVOID"
    NONE = "NONE"

class AnalysisTypeEnum(enum.Enum):
    model = "model"
    human = "human"
    hybrid = "hybrid"

class FeedbackTypeEnum(enum.Enum):
    vote = "vote"
    rating = "rating"
    comment = "comment"

class VoteEnum(enum.Enum):
    agree = "agree"
    disagree = "disagree"
    neutral = "neutral"

class MarketEnum(enum.Enum):
    vn = "vn"
    global_market = "global"  # Database stores as 'global', Python uses global_market

class StrategyHorizonEnum(enum.Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    yearly = "yearly"

class AssetTypeEnum(enum.Enum):
    stocks = "stocks"
    gold = "gold"
    real_estate = "real_estate"

# ------------------------
# Tables
# ------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(255))
    role = Column(SAEnum(RoleEnum), default=RoleEnum.user, nullable=False)
    user_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    feedbacks = relationship("Feedback", back_populates="user")


class News(Base):
    __tablename__ = "news"

    id = Column(Integer, primary_key=True)
    source = Column(String(255), nullable=False)
    url = Column(String(1000), nullable=False, unique=True)
    canonical_url = Column(String(1000), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=False, index=True)
    headline = Column(Text, nullable=False)
    content_raw = Column(Text)                       # optional: for small content
    content_s3_path = Column(String(2000))           # optional: if raw stored on S3/GCS
    content_summary = Column(Text)
    language = Column(String(10), default="vi")
    content_hash = Column(String(128), index=True)  # sha256 for dedupe
    tickers = Column(Text)                          # JSON string of tickers for SQLite compatibility
    asset_type = Column(SAEnum(AssetTypeEnum), default=AssetTypeEnum.stocks, nullable=False)  # Primary asset focus
    tags = Column(JSON)                             # flexible tags/entities
    source_meta = Column(JSON)                      # original source metadata
    is_archived = Column(Boolean, default=False)
    archived_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    market = Column(SAEnum(MarketEnum), default=MarketEnum.global_market, nullable=False)

    analyses = relationship("Analysis", back_populates="news")


class Analysis(Base):
    __tablename__ = "analysis"

    id = Column(Integer, primary_key=True)
    news_id = Column(Integer, ForeignKey("news.id", ondelete="CASCADE"), nullable=False, index=True)
    analysis_type = Column(SAEnum(AnalysisTypeEnum), nullable=False, default=AnalysisTypeEnum.model)
    model_name = Column(String(255))      # e.g. "finbert-sentiment"
    model_version = Column(String(100))   # e.g. "v0.3"
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    sentiment = Column(SAEnum(SentimentEnum))
    sentiment_score = Column(Float)       # -1 .. +1 or 0..1 depending convention
    impact_score = Column(Float)          # 0..1 measure of expected impact
    action_short = Column(SAEnum(ActionEnum))
    action_mid = Column(SAEnum(ActionEnum))
    action_long = Column(SAEnum(ActionEnum))
    action_confidence = Column(Float)     # 0..1
    rationale = Column(Text)              # model/human explanation
    raw_output = Column(JSON)             # raw model JSON output for trace
    is_latest = Column(Boolean, default=True, nullable=False)

    news = relationship("News", back_populates="analyses")
    feedbacks = relationship("Feedback", back_populates="analysis")
    creator = relationship("User", foreign_keys=[created_by])


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    horizon = Column(SAEnum(StrategyHorizonEnum), nullable=False, index=True)
    market = Column(SAEnum(MarketEnum), nullable=False, default=MarketEnum.global_market)
    asset_focus = Column(SAEnum(AssetTypeEnum), nullable=True)  # Optional focus on specific asset
    strategy_date = Column(DateTime(timezone=True), nullable=False, index=True)  # The date this strategy applies to
    
    # Core strategy content
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=False)  # TL;DR section
    key_drivers = Column(JSON, nullable=False)   # Array of 3-5 key market drivers
    action_recommendations = Column(JSON, nullable=False)   # Structured recommendations
    confidence_score = Column(Float)  # 0-1 confidence in this strategy
    
    # Source tracking
    source_analysis_ids = Column(Text, nullable=True)  # JSON string of analysis IDs for SQLite compatibility
    generated_by = Column(String(255))  # Model name/version or "human"
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Ensure one strategy per horizon per market per date
    __table_args__ = (
        UniqueConstraint('horizon', 'market', 'strategy_date', name='uq_strategy_horizon_market_date'),
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("analysis.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    feedback_type = Column(SAEnum(FeedbackTypeEnum), nullable=False, default=FeedbackTypeEnum.vote)
    vote = Column(SAEnum(VoteEnum))
    rating = Column(Integer)              # e.g. 1..5
    selected_action_short = Column(SAEnum(ActionEnum))
    selected_action_mid = Column(SAEnum(ActionEnum))
    selected_action_long = Column(SAEnum(ActionEnum))
    comment = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processed = Column(Boolean, default=False)

    analysis = relationship("Analysis", back_populates="feedbacks")
    user = relationship("User", back_populates="feedbacks")


# ------------------------
# Init DB & create tables + indexes
# ------------------------
def init_db_and_create(uri=None):
    if uri is None:
        uri = os.getenv("DATABASE_URI", "postgresql://postgres:postgres@localhost:5432/finbrief")
    
    engine = create_engine(uri)
    Base.metadata.create_all(engine)

    # Create recommended indexes (PostgreSQL-specific ones only for PostgreSQL)
    with engine.connect() as conn:
        is_postgres = "postgresql" in str(engine.url)
        
        if is_postgres:
            # PostgreSQL-specific extensions and indexes
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
            # Convert JSON columns to JSONB for better performance
            conn.execute(text("ALTER TABLE news ALTER COLUMN tags TYPE jsonb USING tags::jsonb;"))
            conn.execute(text("ALTER TABLE news ALTER COLUMN source_meta TYPE jsonb USING source_meta::jsonb;"))
            conn.execute(text("ALTER TABLE users ALTER COLUMN user_metadata TYPE jsonb USING user_metadata::jsonb;"))
            conn.execute(text("ALTER TABLE strategies ALTER COLUMN key_drivers TYPE jsonb USING key_drivers::jsonb;"))
            conn.execute(text("ALTER TABLE strategies ALTER COLUMN action_recommendations TYPE jsonb USING action_recommendations::jsonb;"))
            
            # GIN indexes for JSONB columns (PostgreSQL only)
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_news_tags_gin ON news USING GIN (tags);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_news_source_meta_gin ON news USING GIN (source_meta);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_strategies_key_drivers_gin ON strategies USING GIN (key_drivers);"))
            
            # Full-text search index (PostgreSQL only)
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_news_search ON news USING GIN (to_tsvector('english', coalesce(headline,'') || ' ' || coalesce(content_summary,'')));"
            ))
        
        # Common indexes that work on both PostgreSQL and SQLite
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_news_content_hash ON news (content_hash);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_news_published_at ON news (published_at);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_strategies_horizon_market ON strategies (horizon, market);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_strategies_strategy_date ON strategies (strategy_date);"))

    print("✅ Tables + recommended indexes are created.")
    return engine


if __name__ == "__main__":
    engine = init_db_and_create()
