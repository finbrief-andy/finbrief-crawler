Main steps:

1. Ingest news from Finnhub: run src/crawlers/ingest_finnhub.py
2. Export training data

- Usage: python src/models/export_training_data.py --out data/training/train.jsonl
- Export training dataset for seq2seq fine-tuning "Action Strategy".
  - Combines news + analysis (+ feedback if available)
  - Outputs JSONL (preferred for HuggingFace datasets) and CSV

3. Train action strategy

- Usage: `PYTHONPATH=. python3 src/models/train_action_strategy.py`
- Fine-tune T5 model to generate Action Strategy
- Input: Headline + Summary + Sentiment
- Output: Short/Mid/Long action strategy
- Model saved to: `models/action-strategy/`

4. Batch inference

- Usage: `PYTHONPATH=. python3 src/models/batch_inference.py`
- Load fine-tuned seq2seq model (T5)
- Fetch news without analysis
- Generate Action Strategy
- Insert into analysis table

5. Store feedback: run store_feedback.py
