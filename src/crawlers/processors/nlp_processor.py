"""
NLP processing components for news analysis.
Handles summarization and sentiment analysis.
"""
import logging
import torch
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class NLPProcessor:
    """Handles NLP tasks: summarization and sentiment analysis"""
    
    def __init__(self):
        self.sent_model = None
        self.tokenizer = None
        self.summarizer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize FinBERT and BART models"""
        SENT_MODEL = "ProsusAI/finbert"
        SUMMARIZER_MODEL = "facebook/bart-large-cnn"
        
        logging.info("Loading FinBERT tokenizer/model (sentiment). This may take time...")
        self.tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL)
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL)
        self.sent_model.eval()
        
        logging.info("Loading summarizer pipeline...")
        self.summarizer = pipeline(
            "summarization", 
            model=SUMMARIZER_MODEL, 
            device=0 if torch.cuda.is_available() else -1
        )
    
    def summarize_text(self, text: str, max_len: int = 80, min_len: int = 25) -> str:
        """Summarize text using BART model"""
        if not text or len(text.split()) < 10:
            return text.strip()
        
        try:
            result = self.summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
            return result[0]["summary_text"]
        except Exception as e:
            logging.warning("Summarizer failed: %s", e)
            return text[:500]
    
    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment using FinBERT.
        Returns: dict with sentiment, confidence, probs, logits
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.sent_model(**inputs)
            logits = outputs.logits.squeeze().cpu().numpy().tolist()
        
        # Apply softmax
        exps = [math.exp(x) for x in logits]
        total = sum(exps)
        probs = [e/total for e in exps]
        
        # FinBERT labels: [negative, neutral, positive]
        labels = ["negative", "neutral", "positive"]
        max_idx = max(range(len(probs)), key=lambda i: probs[i])
        
        return {
            "sentiment": labels[max_idx],
            "confidence": float(probs[max_idx]),
            "probs": probs,
            "logits": logits
        }