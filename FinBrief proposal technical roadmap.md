FinBrief Technical Overview (RSS/API-first)

1. Architecture Overview

Flow:
RSS/API Feeds → Ingestion Worker → Database → Processing Pipeline (clean, dedupe, categorize, sentiment, embeddings) → Strategy Generator (LLM) → API + UI Delivery

This trims down complexity: no Playwright/scraping infra needed at MVP stage.

2. Components
   A. Ingestion Layer

Sources

Vietnam: VnExpress (business RSS), Vietstock RSS, CafeF RSS (if available).

Global: CNBC RSS, MarketWatch RSS.

API: Finnhub.io (/api/v1/news) for structured market news.

Fetcher

Schedule worker (Celery/CRON).

Library: feedparser (Python) for RSS, requests for JSON APIs.

Storage

Database (Postgres) with schema for articles:

articles (
id uuid PRIMARY KEY,
source varchar,
title text,
summary text,
link text,
published_at timestamptz,
content text,
tickers text[],
sentiment float,
inserted_at timestamptz default now()
)

Optional: raw XML/JSON logs stored in S3 for replay/debug.

B. Processing Layer

Deduplication

Hash (title + published_at) to prevent duplicates.

Categorization

Tag by asset_type (stocks, gold, real estate).

Map tickers via regex/dictionaries for VN30 + S&P500 symbols.

Sentiment Analysis

Lightweight models:

English: Hugging Face finbert.

Vietnamese: PhoBERT sentiment classifier.

Embeddings

Generate embeddings for summary + title.

Vector DB options: Chroma (simple/local) or Qdrant (production-ready).

C. Strategy Generation Layer

Retrieval

Pull top N articles (by recency + sentiment weight + relevance to tickers).

Summarization & Action Strategy

LLM pipeline (OpenAI GPT / local LLaMA/Mistral with RAG).

System prompt enforces structure:

TL;DR

Key Drivers (3–5 bullets)

Action Recommendations (buy/hold/sell/hedge/reallocate) with rationale

Source links for transparency

Strategy Storage

Table: strategies (id, horizon, market, json_payload, generated_at)

D. Delivery Layer

API (FastAPI)

GET /strategy/daily → latest daily JSON

GET /strategy/weekly → weekly JSON

GET /articles/:id → raw/refined article

Web Dashboard (React/Tailwind)

Card view for each horizon (daily/weekly/monthly/yearly).

Expand to show linked articles.

Email/Telegram Bot (Phase 2)

Push “Today’s Strategy Page” each morning.

3. Data Flow Example

Worker fetches VnExpress RSS at 7:00 → inserts 20 articles.

Processing tags 5 with tickers (VCB, VIC).

Sentiment scores applied → embeddings stored.

At 8:00, Strategy Generator retrieves top 10, summarizes into JSON.

API publishes /strategy/daily.

User opens app → sees 1-page strategy with action recommendations.

4. MVP Development Roadmap

Week 1–2

Setup repo, Postgres, FastAPI skeleton.

Ingest VN RSS feeds (VnExpress, Vietstock).

Store to DB.

Week 3–4

Add dedupe + categorization.

Integrate sentiment + embeddings (vector DB).

Build simple retrieval API.

Week 5–6

Build LLM summarizer → output daily strategy JSON.

Build basic web UI card.

Week 7–8 (MVP Launch)

Publish daily strategies to small test group (email + dashboard).

Collect feedback on clarity, trust, usefulness.

5. Scaling Later

Add full-text extraction (scrape article link) for deeper context.

Add personalization (filter by ticker/sector).

Add alerts (unusual moves, institutional trades).

Expand sources (Bloomberg-lite experience).
