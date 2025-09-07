# models_migration.py
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Boolean, Float,
    Enum as SAEnum, ForeignKey, func, UniqueConstraint, text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
import enum
import os

Base = declarative_base()

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
    user_metadata = Column(JSONB, nullable=True)
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
    tickers = Column(ARRAY(String))                 # e.g. ['VIC', 'VCB']
    tags = Column(JSONB)                            # flexible tags/entities
    source_meta = Column(JSONB)                     # original source metadata
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
    raw_output = Column(JSONB)            # raw model JSON output for trace
    is_latest = Column(Boolean, default=True, nullable=False)

    news = relationship("News", back_populates="analyses")
    feedbacks = relationship("Feedback", back_populates="analysis")
    creator = relationship("User", foreign_keys=[created_by])


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

    # Create recommended Postgres indexes not expressed in SQLAlchemy models (full-text / GIN)
    with engine.connect() as conn:
        # Enable pg_trgm if you want fuzzy matching (optional)
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
        # GIN index for tickers array
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_news_tickers_gin ON news USING GIN (tickers);"))
        # GIN index for tags jsonb
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_news_tags_gin ON news USING GIN (tags);"))
        # Full-text search index on headline + content_summary
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_news_search ON news USING GIN (to_tsvector('simple', coalesce(headline,'') || ' ' || coalesce(content_summary,'')));"
        ))
        # Index for content_hash for dedupe fast lookup
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_news_content_hash ON news (content_hash);"))
        # Index on published_at already created via Column(index=True), but ensure btree index exists
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_news_published_at ON news (published_at);"))

    print("✅ Tables + recommended indexes are created.")
    return engine


if __name__ == "__main__":
    engine = init_db_and_create()
