# export_training_data.py
"""
Export training dataset for seq2seq fine-tuning "Action Strategy".

- Combines news + analysis (+ feedback if available)
- Outputs JSONL (preferred for HuggingFace datasets) and CSV

Usage:
  python export_training_data.py --out data/train.jsonl
"""

import os
import json
import argparse
import pandas as pd
from sqlalchemy.orm import sessionmaker
from src.database.models_migration import init_db_and_create, News, Analysis, Feedback, ActionEnum

DATABASE_URI = os.getenv("DATABASE_URI")

def export_dataset(out_jsonl="train.jsonl", out_csv="train.csv"):
    # Create data/training folder if it doesn't exist
    export_dir = "data/training"
    os.makedirs(export_dir, exist_ok=True)
    
    # Update file paths to include data/training folder
    out_jsonl = os.path.join(export_dir, os.path.basename(out_jsonl))
    out_csv = os.path.join(export_dir, os.path.basename(out_csv))
    
    engine = init_db_and_create(DATABASE_URI)
    Session = sessionmaker(bind=engine)
    session = Session()

    dataset = []

    # Query join news + analysis (latest only)
    q = (
        session.query(News, Analysis)
        .join(Analysis, News.id == Analysis.news_id)
        .filter(Analysis.is_latest == True)
        .all()
    )

    for news, analysis in q:
        # Base input
        input_text = f"Headline: {news.headline}\n"
        if news.content_summary:
            input_text += f"Summary: {news.content_summary}\n"
        if analysis.sentiment:
            input_text += f"Sentiment: {analysis.sentiment.value}\n"

        # Default output (from analysis)
        output_text = f"Short={analysis.action_short.value if analysis.action_short else 'NONE'}, "
        output_text += f"Mid={analysis.action_mid.value if analysis.action_mid else 'NONE'}, "
        output_text += f"Long={analysis.action_long.value if analysis.action_long else 'NONE'}"

        # Check feedback for override
        fb = (
            session.query(Feedback)
            .filter(Feedback.analysis_id == analysis.id)
            .first()
        )
        if fb:
            # If user selected actions exist, override
            if fb.selected_action_short or fb.selected_action_mid or fb.selected_action_long:
                output_text = f"Short={fb.selected_action_short.value if fb.selected_action_short else 'NONE'}, "
                output_text += f"Mid={fb.selected_action_mid.value if fb.selected_action_mid else 'NONE'}, "
                output_text += f"Long={fb.selected_action_long.value if fb.selected_action_long else 'NONE'}"
            # Optionally, enrich input with comment
            if fb.comment:
                input_text += f"\nUserComment: {fb.comment}\n"

        dataset.append({"input": input_text.strip(), "output": output_text.strip()})

    session.close()

    # Write JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write CSV
    df = pd.DataFrame(dataset)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"✅ Exported {len(dataset)} samples to {out_jsonl} and {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="train.jsonl", help="Output JSONL file")
    args = parser.parse_args()
    export_dataset(out_jsonl=args.out, out_csv=args.out.replace(".jsonl", ".csv"))
