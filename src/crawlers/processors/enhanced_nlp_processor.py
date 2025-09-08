"""
Enhanced NLP processing components for advanced news analysis.
Includes Named Entity Recognition, improved summarization, sentiment analysis, and key phrase extraction.
"""
import logging
import torch
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import warnings

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoModelForTokenClassification, pipeline, 
    BertTokenizer, BertModel
)

# Suppress warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning)


class EnhancedNLPProcessor:
    """Advanced NLP processor with NER, summarization, sentiment analysis, and key phrase extraction"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device set to use {self.device}")
        
        # Model components
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.ner_pipeline = None
        self.summarizer = None
        self.bert_model = None
        self.bert_tokenizer = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all NLP models"""
        try:
            self._initialize_sentiment_model()
            self._initialize_ner_model()
            self._initialize_summarizer()
            self._initialize_bert_model()
        except Exception as e:
            logging.error(f"Error initializing NLP models: {e}")
            # Continue with limited functionality
    
    def _initialize_sentiment_model(self):
        """Initialize FinBERT for financial sentiment analysis"""
        try:
            model_name = "ProsusAI/finbert"
            logging.info("Loading FinBERT for sentiment analysis...")
            
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_model.eval()
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.sentiment_model = self.sentiment_model.to(self.device)
            
            logging.info("✅ FinBERT sentiment model loaded")
        except Exception as e:
            logging.warning(f"Could not load sentiment model: {e}")
    
    def _initialize_ner_model(self):
        """Initialize NER model for entity recognition"""
        try:
            logging.info("Loading NER model...")
            # Use a financial/business-oriented NER model or general English NER
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
            logging.info("✅ NER model loaded")
        except Exception as e:
            logging.warning(f"Could not load NER model: {e}")
    
    def _initialize_summarizer(self):
        """Initialize enhanced summarization model"""
        try:
            logging.info("Loading summarization model...")
            # Use BART for better summarization
            model_name = "facebook/bart-large-cnn"
            
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            logging.info("✅ Summarization model loaded")
        except Exception as e:
            logging.warning(f"Could not load summarization model: {e}")
    
    def _initialize_bert_model(self):
        """Initialize BERT model for embeddings and key phrase extraction"""
        try:
            logging.info("Loading BERT model for embeddings...")
            model_name = "bert-base-uncased"
            
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert_model = BertModel.from_pretrained(model_name)
            self.bert_model.eval()
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.bert_model = self.bert_model.to(self.device)
            
            logging.info("✅ BERT model loaded")
        except Exception as e:
            logging.warning(f"Could not load BERT model: {e}")
    
    def extract_named_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities (companies, people, locations) from text
        
        Returns:
            Dict with entity categories and their details
        """
        if not self.ner_pipeline:
            return {"entities": [], "companies": [], "people": [], "locations": []}
        
        try:
            # Get raw NER results
            entities = self.ner_pipeline(text)
            
            # Categorize entities
            companies = []
            people = []
            locations = []
            other_entities = []
            
            for entity in entities:
                entity_info = {
                    "text": entity["word"],
                    "label": entity["entity_group"],
                    "confidence": entity["score"],
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0)
                }
                
                # Categorize based on NER labels
                if entity["entity_group"] in ["ORG"]:
                    companies.append(entity_info)
                elif entity["entity_group"] in ["PER"]:
                    people.append(entity_info)
                elif entity["entity_group"] in ["LOC", "GPE"]:
                    locations.append(entity_info)
                else:
                    other_entities.append(entity_info)
            
            # Post-process: try to identify financial companies from context
            companies.extend(self._extract_financial_companies(text))
            
            return {
                "entities": entities,
                "companies": companies,
                "people": people,
                "locations": locations,
                "other": other_entities
            }
            
        except Exception as e:
            logging.error(f"NER extraction failed: {e}")
            return {"entities": [], "companies": [], "people": [], "locations": []}
    
    def _extract_financial_companies(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial companies using rule-based patterns"""
        companies = []
        
        # Common financial company patterns
        patterns = [
            r'\b([A-Z][a-z]+ & [A-Z][a-z]+)\b',  # Company & Company
            r'\b([A-Z][a-z]+ Inc\.?)\b',          # Company Inc
            r'\b([A-Z][a-z]+ Corp\.?)\b',         # Company Corp
            r'\b([A-Z][a-z]+ Ltd\.?)\b',          # Company Ltd
            r'\b([A-Z][a-z]+ Group)\b',           # Company Group
            r'\b([A-Z][a-z]+ Bank)\b',            # Company Bank
            r'\b([A-Z]+)\s+\([A-Z]{2,5}\)\b',     # COMPANY (TICKER)
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                companies.append({
                    "text": match.group(1),
                    "label": "COMPANY",
                    "confidence": 0.8,  # Rule-based confidence
                    "start": match.start(),
                    "end": match.end(),
                    "source": "rule_based"
                })
        
        return companies
    
    def enhanced_summarize(self, text: str, max_length: int = 100, min_length: int = 30) -> Dict[str, Any]:
        """
        Enhanced text summarization with multiple strategies
        
        Returns:
            Dict with summary, method used, and metadata
        """
        if not text or len(text.split()) < 10:
            return {
                "summary": text.strip(),
                "method": "original",
                "confidence": 1.0,
                "word_count": len(text.split())
            }
        
        try:
            # Strategy 1: Use BART summarization for longer texts
            if self.summarizer and len(text.split()) > 50:
                result = self.summarizer(
                    text, 
                    max_length=max_length, 
                    min_length=min_length, 
                    do_sample=False
                )
                
                return {
                    "summary": result[0]["summary_text"],
                    "method": "bart_transformer",
                    "confidence": 0.9,
                    "word_count": len(result[0]["summary_text"].split()),
                    "original_length": len(text.split())
                }
            
            # Strategy 2: Extractive summarization for medium texts
            elif len(text.split()) > 20:
                summary = self._extractive_summarization(text, max_sentences=3)
                return {
                    "summary": summary,
                    "method": "extractive",
                    "confidence": 0.8,
                    "word_count": len(summary.split()),
                    "original_length": len(text.split())
                }
            
            # Strategy 3: First sentence + key info for short texts
            else:
                sentences = text.split('. ')
                summary = sentences[0]
                if len(sentences) > 1 and len(summary.split()) < 15:
                    summary += '. ' + sentences[1]
                
                return {
                    "summary": summary,
                    "method": "first_sentences",
                    "confidence": 0.7,
                    "word_count": len(summary.split()),
                    "original_length": len(text.split())
                }
                
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            # Fallback: truncate to first 100 words
            words = text.split()[:100]
            return {
                "summary": ' '.join(words),
                "method": "truncation",
                "confidence": 0.5,
                "word_count": len(words),
                "error": str(e)
            }
    
    def _extractive_summarization(self, text: str, max_sentences: int = 3) -> str:
        """Simple extractive summarization using sentence scoring"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences by word frequency and position
        word_freq = Counter()
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            word_freq.update(words)
        
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words = re.findall(r'\b\w+\b', sentence.lower())
            score = sum(word_freq[word] for word in words)
            # Boost score for early sentences
            score *= (1 + 0.5 / (i + 1))
            sentence_scores.append((score, sentence))
        
        # Select top sentences
        top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
        # Maintain original order
        selected = [s for _, s in sorted(top_sentences, key=lambda x: sentences.index(x[1]))]
        
        return '. '.join(selected) + '.'
    
    def enhanced_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Enhanced sentiment analysis with confidence and context
        
        Returns:
            Dict with sentiment, confidence, probabilities, and context
        """
        if not self.sentiment_model:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "method": "fallback",
                "error": "Model not available"
            }
        
        try:
            # Tokenize and get model predictions
            inputs = self.sentiment_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                logits = outputs.logits.squeeze().cpu().numpy()
            
            # Apply softmax to get probabilities
            exp_logits = torch.exp(torch.tensor(logits))
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
            
            # FinBERT labels: [negative, neutral, positive]
            labels = ["negative", "neutral", "positive"]
            max_idx = probs.argmax()
            
            # Additional analysis
            sentiment_strength = self._calculate_sentiment_strength(probs)
            market_implications = self._analyze_market_implications(text, labels[max_idx])
            
            return {
                "sentiment": labels[max_idx],
                "confidence": float(probs[max_idx]),
                "probabilities": {
                    "negative": float(probs[0]),
                    "neutral": float(probs[1]), 
                    "positive": float(probs[2])
                },
                "strength": sentiment_strength,
                "market_implications": market_implications,
                "method": "finbert_enhanced",
                "logits": logits.tolist()
            }
            
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "method": "error_fallback",
                "error": str(e)
            }
    
    def _calculate_sentiment_strength(self, probs) -> str:
        """Calculate sentiment strength based on probability distribution"""
        max_prob = max(probs)
        
        if max_prob > 0.8:
            return "strong"
        elif max_prob > 0.6:
            return "moderate"
        else:
            return "weak"
    
    def _analyze_market_implications(self, text: str, sentiment: str) -> List[str]:
        """Analyze potential market implications based on content and sentiment"""
        implications = []
        
        # Financial keywords that amplify implications
        financial_keywords = {
            'earnings': ['revenue_impact', 'profit_expectations'],
            'revenue': ['growth_prospects', 'market_share'],
            'profit': ['investor_confidence', 'dividend_potential'],
            'loss': ['risk_assessment', 'recovery_timeline'],
            'acquisition': ['market_consolidation', 'competitive_advantage'],
            'merger': ['market_dynamics', 'regulatory_review'],
            'ipo': ['market_expansion', 'valuation_assessment'],
            'bankruptcy': ['sector_impact', 'contagion_risk'],
            'partnership': ['strategic_synergy', 'market_access'],
            'lawsuit': ['regulatory_risk', 'financial_liability']
        }
        
        text_lower = text.lower()
        for keyword, impacts in financial_keywords.items():
            if keyword in text_lower:
                implications.extend(impacts)
        
        # Sentiment-based implications
        if sentiment == "positive":
            implications.extend(['potential_upside', 'investor_optimism'])
        elif sentiment == "negative":
            implications.extend(['potential_downside', 'investor_caution'])
        
        return list(set(implications))  # Remove duplicates
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[Dict[str, Any]]:
        """
        Extract key phrases using multiple techniques
        
        Returns:
            List of key phrases with scores and methods
        """
        key_phrases = []
        
        try:
            # Method 1: Statistical approach (TF-IDF like)
            statistical_phrases = self._extract_statistical_phrases(text)
            key_phrases.extend(statistical_phrases)
            
            # Method 2: Named entities as key phrases
            entities = self.extract_named_entities(text)
            for category in ['companies', 'people', 'locations']:
                for entity in entities[category]:
                    key_phrases.append({
                        "phrase": entity["text"],
                        "score": entity["confidence"],
                        "method": f"ner_{category}",
                        "category": category
                    })
            
            # Method 3: Financial terms
            financial_phrases = self._extract_financial_terms(text)
            key_phrases.extend(financial_phrases)
            
            # Deduplicate and sort by score
            seen = set()
            unique_phrases = []
            for phrase in key_phrases:
                if phrase["phrase"].lower() not in seen:
                    seen.add(phrase["phrase"].lower())
                    unique_phrases.append(phrase)
            
            # Sort by score and limit results
            unique_phrases.sort(key=lambda x: x["score"], reverse=True)
            return unique_phrases[:max_phrases]
            
        except Exception as e:
            logging.error(f"Key phrase extraction failed: {e}")
            return []
    
    def _extract_statistical_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Extract key phrases using statistical methods"""
        phrases = []
        
        # Extract n-grams (2-4 words)
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Extract bigrams and trigrams
        for n in [2, 3, 4]:
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            ngram_freq = Counter(ngrams)
            
            for ngram, freq in ngram_freq.most_common(10):
                # Filter out common stopword combinations
                if not self._is_stopword_phrase(ngram) and freq > 1:
                    score = freq / len(ngrams)  # Normalized frequency
                    phrases.append({
                        "phrase": ngram,
                        "score": score,
                        "method": f"statistical_ngram_{n}",
                        "frequency": freq
                    })
        
        return phrases
    
    def _extract_financial_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial-specific terms and phrases"""
        phrases = []
        
        # Financial term patterns
        financial_patterns = [
            (r'\$[\d,]+\.?\d*[BMK]?', 'monetary_amount'),
            (r'\b\d+\.?\d*%', 'percentage'),
            (r'\b(?:Q[1-4]|quarter)\s+(?:earnings|results)', 'earnings_period'),
            (r'\b(?:revenue|profit|loss|earnings|EBITDA)\b', 'financial_metric'),
            (r'\b(?:IPO|merger|acquisition|partnership)\b', 'corporate_action'),
            (r'\b(?:bull|bear)\s+market\b', 'market_condition'),
            (r'\b(?:dividend|yield|ROI|ROE)\b', 'investment_metric')
        ]
        
        for pattern, category in financial_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrases.append({
                    "phrase": match.group(),
                    "score": 0.8,  # High score for financial terms
                    "method": "financial_pattern",
                    "category": category,
                    "position": match.start()
                })
        
        return phrases
    
    def _is_stopword_phrase(self, phrase: str) -> bool:
        """Check if phrase consists mainly of stopwords"""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        words = phrase.split()
        stopword_count = sum(1 for word in words if word in stopwords)
        return stopword_count >= len(words) * 0.7  # More than 70% stopwords
    
    def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive NLP analysis combining all techniques
        
        Returns:
            Dict with all analysis results
        """
        try:
            logging.info("Starting comprehensive NLP analysis...")
            
            # Perform all analyses
            entities = self.extract_named_entities(text)
            summary = self.enhanced_summarize(text)
            sentiment = self.enhanced_sentiment_analysis(text)
            key_phrases = self.extract_key_phrases(text)
            
            # Calculate overall insights
            insights = self._generate_insights(entities, sentiment, key_phrases)
            
            return {
                "entities": entities,
                "summary": summary,
                "sentiment": sentiment,
                "key_phrases": key_phrases,
                "insights": insights,
                "metadata": {
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "processing_time": "calculated_elsewhere",
                    "models_used": self._get_active_models()
                }
            }
            
        except Exception as e:
            logging.error(f"Comprehensive analysis failed: {e}")
            return {
                "error": str(e),
                "text_length": len(text) if text else 0,
                "fallback": True
            }
    
    def _generate_insights(self, entities: Dict, sentiment: Dict, key_phrases: List) -> Dict[str, Any]:
        """Generate high-level insights from analysis results"""
        insights = {
            "market_relevance": "medium",
            "entity_diversity": len(entities.get("companies", [])) + len(entities.get("people", [])),
            "sentiment_clarity": sentiment.get("confidence", 0.5),
            "key_theme_count": len(key_phrases),
            "recommendations": []
        }
        
        # Market relevance based on entities and key phrases
        company_count = len(entities.get("companies", []))
        financial_phrases = len([p for p in key_phrases if p.get("category") in ["financial_metric", "corporate_action"]])
        
        if company_count >= 2 or financial_phrases >= 3:
            insights["market_relevance"] = "high"
        elif company_count >= 1 or financial_phrases >= 1:
            insights["market_relevance"] = "medium"
        else:
            insights["market_relevance"] = "low"
        
        # Recommendations based on analysis
        if sentiment.get("confidence", 0) > 0.8:
            insights["recommendations"].append("High confidence sentiment - suitable for trading signals")
        
        if insights["entity_diversity"] > 3:
            insights["recommendations"].append("Multiple entities detected - good for relationship analysis")
        
        if financial_phrases > 0:
            insights["recommendations"].append("Contains financial metrics - suitable for quantitative analysis")
        
        return insights
    
    def _get_active_models(self) -> List[str]:
        """Get list of successfully loaded models"""
        models = []
        if self.sentiment_model:
            models.append("finbert_sentiment")
        if self.ner_pipeline:
            models.append("bert_ner")
        if self.summarizer:
            models.append("bart_summarizer")
        if self.bert_model:
            models.append("bert_embeddings")
        return models