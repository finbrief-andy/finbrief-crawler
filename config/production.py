"""
Production Configuration for FinBrief
"""
import os
from urllib.parse import quote_plus

# Database Configuration
DATABASE_USER = os.getenv("DB_USER", "andy.huynh")
DATABASE_PASSWORD = os.getenv("DB_PASSWORD", "")
DATABASE_HOST = os.getenv("DB_HOST", "localhost")
DATABASE_PORT = os.getenv("DB_PORT", "5432")
DATABASE_NAME = os.getenv("DB_NAME", "finbrief_prod")

# Construct PostgreSQL URI
if DATABASE_PASSWORD:
    DATABASE_URI = f"postgresql://{DATABASE_USER}:{quote_plus(DATABASE_PASSWORD)}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
else:
    DATABASE_URI = f"postgresql://{DATABASE_USER}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# Test database URI
TEST_DATABASE_URI = os.getenv("TEST_DATABASE_URI", "sqlite:///./test_finbrief.db")

# API Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "finbrief-production-secret-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# News Sources Configuration
NEWS_SOURCES = {
    "bloomberg_rss": "https://feeds.bloomberg.com/markets/news.rss",
    "cnbc_rss": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "marketwatch_rss": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews"
}

# Processing Configuration
MAX_ARTICLES_PER_SOURCE = 50
DUPLICATE_THRESHOLD = 0.85
SENTIMENT_THRESHOLD = 0.6

# Vector Store Configuration
VECTOR_STORE_ENABLED = True
VECTOR_STORE_COLLECTION = "finbrief_news"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"