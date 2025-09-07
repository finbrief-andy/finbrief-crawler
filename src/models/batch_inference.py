# batch_inference.py
"""
Batch inference:
- Load fine-tuned seq2seq model (T5)
- Fetch news
- If news has old analysis -> mark is_latest=False
- Insert new analysis with model output
"""

import os
import torch
from sqlalchemy.orm import sessionmaker
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.database.models_migration import init_db_and_create, News, Analysis, ActionEnum

# ===== CONFIG =====
DATABASE_URI = os.getenv("DATABASE_URI")
MODEL_DIR = "./models/action-strategy"

# ===== LOAD MODEL =====
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

if torch.cuda.is_available():
    model = model.to("cuda")

def generate_action_strategy(headline, summary, sentiment):
    input_text = f"Headline: {headline}\nSummary: {summary}\nSentiment: {sentiment}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def parse_strategy(output_text):
    """
    Convert model text into ActionEnum values.
    Example: "Short=SELL, Mid=HOLD, Long=BUY"
    """
    action_map = {"BUY": ActionEnum.BUY, "SELL": ActionEnum.SELL, "HOLD": ActionEnum.HOLD}
    short, mid, long_ = None, None, None

    parts = output_text.replace(" ", "").split(",")
    for p in parts:
        if p.startswith("Short="):
            short = action_map.get(p.split("=")[1].upper(), None)
        elif p.startswith("Mid="):
            mid = action_map.get(p.split("=")[1].upper(), None)
        elif p.startswith("Long="):
            long_ = action_map.get(p.split("=")[1].upper(), None)

    return short, mid, long_

def main():
    engine = init_db_and_create(DATABASE_URI)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Fetch batch of news
    news_list = session.query(News).limit(20).all()
    print(f"🔎 Found {len(news_list)} news to process")

    for news in news_list:
        if not news.content_summary or not news.headline:
            continue

        sentiment = "neutral"  # fallback nếu chưa có sentiment
        # có thể chạy lại FinBERT ở đây nếu muốn sentiment mới

        # Run model
        output_text = generate_action_strategy(news.headline, news.content_summary, sentiment)
        short, mid, long_ = parse_strategy(output_text)

        # Mark old analysis as not latest
        old_analyses = session.query(Analysis).filter(Analysis.news_id == news.id, Analysis.is_latest == True).all()
        for old in old_analyses:
            old.is_latest = False
            session.add(old)

        # Insert new analysis
        new_analysis = Analysis(
            news_id=news.id,
            sentiment=sentiment,
            impact_score=None,
            action_short=short,
            action_mid=mid,
            action_long=long_,
            model_version="t5-finetuned-v1",
            is_latest=True
        )
        session.add(new_analysis)
        session.commit()
        print(f"✅ Updated analysis for news_id={news.id}: {output_text}")

    session.close()

if __name__ == "__main__":
    main()
