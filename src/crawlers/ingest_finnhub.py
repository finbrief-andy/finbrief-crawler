# ingest_finnhub.py
"""
Ingest pipeline:
  Finnhub -> dedupe -> store news -> summarize -> sentiment (FinBERT) -> store analysis

Pre-req:
  - models_migration.py (schema + init_db_and_create) must be in same folder or installed module
  - Set env vars: FINNHUB_API_KEY, DATABASE_URI
  - Optional (for S3): AWS_S3_BUCKET, AWS credentials
"""

import os
import requests
import hashlib
import json
import time
import logging
from datetime import datetime

from sqlalchemy.orm import sessionmaker

# transformers & torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# import DB models & init func from your migration file
# Import from database module
from src.database.models_migration import (
    init_db_and_create, News, Analysis, User,
    SentimentEnum, ActionEnum, AnalysisTypeEnum
)

# Optional S3
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    S3_AVAILABLE = True
except Exception:
    S3_AVAILABLE = False

# Config / env
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d2u2v59r01qo4hodrjagd2u2v59r01qo4hodrjb0")
DATABASE_URI = os.getenv("DATABASE_URI")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")  # optional
UPLOAD_TO_S3 = bool(AWS_S3_BUCKET) and S3_AVAILABLE

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if not FINNHUB_API_KEY:
    logging.error("Please set FINNHUB_API_KEY env variable.")
    raise SystemExit(1)

# Initialize DB
engine = init_db_and_create(DATABASE_URI)
Session = sessionmaker(bind=engine)

# Initialize models for NLP
SENT_MODEL = "ProsusAI/finbert"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"

logging.info("Loading FinBERT tokenizer/model (sentiment). This may take time...")
tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL)
sent_model.eval()

logging.info("Loading summarizer pipeline...")
summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device=0 if torch.cuda.is_available() else -1)

# Optional S3 client
s3_client = None
if UPLOAD_TO_S3:
    if not S3_AVAILABLE:
        logging.error("boto3 not available; disable S3 upload.")
        UPLOAD_TO_S3 = False
    else:
        s3_client = boto3.client("s3")
        logging.info(f"S3 upload enabled, bucket: {AWS_S3_BUCKET}")

# --- Helpers ---
def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fetch_finnhub_news(category="general", min_id=None):
    """
    Fetch news list from Finnhub.
    - category: general | forex | crypto | mergers | ipo | ...
    Returns list of dicts as Finnhub provides.
    """
    url = f"https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_API_KEY}"
    logging.info(f"Fetching Finnhub news category={category}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data  # list of news items

def summarize_text(text: str, max_len=80, min_len=25):
    # handle very short text
    if not text or len(text.split()) < 10:
        return text.strip()
    # HuggingFace summarizer returns list
    try:
        out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        logging.warning("Summarizer failed: %s", e)
        return text[:500]

def analyze_sentiment_finbert(text: str):
    """
    Return: dict {sentiment: 'positive'|'neutral'|'negative', confidence: float, raw_logits: [...]}.
    FinBERT output ordering is [negative, neutral, positive] (model dependent).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sent_model(**inputs)
        logits = outputs.logits.squeeze().cpu().numpy().tolist()
    # softmax
    import math
    exps = [math.exp(x) for x in logits]
    s = sum(exps)
    probs = [e/s for e in exps]
    # mapping as ProsusAI/finbert: labels [negative, neutral, positive]
    labels = ["negative", "neutral", "positive"]
    idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {"sentiment": labels[idx], "confidence": float(probs[idx]), "probs": probs, "logits": logits}

def s3_upload_text(bucket, key, text):
    if not s3_client:
        raise RuntimeError("S3 client not initialized")
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
        return f"s3://{bucket}/{key}"
    except (BotoCoreError, ClientError) as e:
        logging.error("S3 upload failed: %s", e)
        return None

def map_sentiment_to_action(sentiment_label: str):
    """
    Simple rule-based mapping: adjust later.
    Returns actions dict with short/mid/long keys (ActionEnum).
    """
    if sentiment_label == "positive":
        return {"short": ActionEnum.BUY, "mid": ActionEnum.HOLD, "long": ActionEnum.BUY}
    elif sentiment_label == "negative":
        return {"short": ActionEnum.SELL, "mid": ActionEnum.AVOID, "long": ActionEnum.HOLD}
    else:  # neutral
        return {"short": ActionEnum.HOLD, "mid": ActionEnum.HOLD, "long": ActionEnum.HOLD}

# --- Main ingest function ---
def ingest_news_batch(category="general"):
    items = fetch_finnhub_news(category=category)
    session = Session()
    inserted = 0
    skipped = 0

    for item in items:
        try:
            # Finnhub fields: category, datetime (unix), headline, summary, url, image?
            headline = item.get("headline") or item.get("title") or ""
            summary_text = item.get("summary") or ""
            url = item.get("url") or item.get("news_url") or None
            published_unix = item.get("datetime")  # seconds
            published_at = datetime.utcfromtimestamp(published_unix) if published_unix else datetime.utcnow()
            source = item.get("source") or item.get("category") or "finnhub"

            # Compose content for hash & summarization
            content_for_hash = (headline + " " + summary_text).strip()
            content_hash = sha256_hash(content_for_hash)

            # Dedupe by URL or content_hash
            existing = session.query(News).filter(
                (News.url == url) | (News.content_hash == content_hash)
            ).first()
            if existing:
                logging.info("Skipping duplicate news: %s", headline[:80])
                skipped += 1
                continue

            # Optionally upload raw content to S3 if enabled and content large
            content_raw = item.get("summary") or item.get("description") or ""
            content_s3_path = None
            if UPLOAD_TO_S3 and content_raw:
                key = f"news/{published_at.strftime('%Y/%m/%d')}/{content_hash}.txt"
                s3_path = s3_upload_text(AWS_S3_BUCKET, key, content_raw)
                if s3_path:
                    content_s3_path = s3_path
                    # To save DB space, we don't store content_raw when s3 used
                    content_raw_db = None
                else:
                    content_raw_db = content_raw
            else:
                content_raw_db = content_raw

            # Summarize (use headline+summary or content)
            text_to_summarize = content_raw_db or (headline + ". " + summary_text)
            content_summary = summarize_text(text_to_summarize, max_len=80, min_len=20)

            # Insert news
            news = News(
                source=source,
                url=url,
                published_at=published_at,
                headline=headline,
                content_raw=content_raw_db,
                content_s3_path=content_s3_path,
                content_summary=content_summary,
                content_hash=content_hash,
                # language default in model is 'vi' but Finnhub items usually en -> let default
            )
            session.add(news)
            session.commit()
            logging.info("Inserted news id=%s url=%s", news.id, url)

            # Analyze sentiment
            # Choose text for sentiment: prefer summary (concise) or headline+summary
            text_for_sent = (headline + ". " + content_summary) if content_summary else headline
            sent = analyze_sentiment_finbert(text_for_sent)

            # Map to action
            actions = map_sentiment_to_action(sent["sentiment"])

            # raw_output: keep both sentiment model output and summarizer meta
            raw_output = {
                "finnhub_item": item,
                "summarizer": {"summary": content_summary},
                "sentiment_model": sent
            }

            # Insert analysis (model-based)
            analysis = Analysis(
                news_id=news.id,
                analysis_type=AnalysisTypeEnum.model,
                model_name="finbert+bart",
                model_version="v0.1",
                created_by=None,
                sentiment=SentimentEnum(sent["sentiment"]),
                sentiment_score=sent["confidence"],
                impact_score=None,
                action_short=actions["short"],
                action_mid=actions["mid"],
                action_long=actions["long"],
                action_confidence=float(sent["confidence"]),
                rationale=f"Auto rule-based mapping from sentiment ({sent['sentiment']})",
                raw_output=raw_output,
                is_latest=True
            )
            session.add(analysis)
            session.commit()
            logging.info("Inserted analysis id=%s for news id=%s (sentiment=%s)", analysis.id, news.id, sent["sentiment"])
            inserted += 1

            # Small sleep to be polite to APIs
            time.sleep(0.2)

        except Exception as e:
            logging.exception("Error processing item: %s", e)
            session.rollback()
    session.close()
    logging.info("Done. Inserted=%d, Skipped=%d", inserted, skipped)
    return inserted, skipped

# === Run if script executed ===
if __name__ == "__main__":
    # Example: ingest 'general' news
    ingest_news_batch("general")
