"""
NLP Processor Adapter - Provides backward compatibility and enhanced features.
Falls back to original processor if enhanced models fail to load.
"""
import logging
from typing import Dict, Any, Optional


class NLPProcessorAdapter:
    """
    Adapter that provides enhanced NLP features with fallback to original processor.
    Ensures backward compatibility while enabling advanced features when available.
    """
    
    def __init__(self):
        self.enhanced_processor = None
        self.original_processor = None
        self.use_enhanced = False
        
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize processors with fallback strategy"""
        # Try to load enhanced processor first
        try:
            from .enhanced_nlp_processor import EnhancedNLPProcessor
            self.enhanced_processor = EnhancedNLPProcessor()
            self.use_enhanced = True
            logging.info("✅ Enhanced NLP processor loaded successfully")
        except Exception as e:
            logging.warning(f"Enhanced NLP processor failed to load: {e}")
            self.use_enhanced = False
        
        # Always load original processor as fallback
        try:
            from .nlp_processor import NLPProcessor
            self.original_processor = NLPProcessor()
            logging.info("✅ Original NLP processor loaded as fallback")
        except Exception as e:
            logging.error(f"Original NLP processor failed to load: {e}")
            raise RuntimeError("No NLP processor available")
    
    def summarize_text(self, text: str, max_len: int = 80, min_len: int = 25) -> str:
        """
        Summarize text using best available method.
        Falls back to original processor if enhanced fails.
        """
        if self.use_enhanced and self.enhanced_processor:
            try:
                result = self.enhanced_processor.enhanced_summarize(
                    text, max_length=max_len, min_length=min_len
                )
                return result.get('summary', text[:500])
            except Exception as e:
                logging.warning(f"Enhanced summarization failed, using fallback: {e}")
        
        # Fallback to original processor
        if self.original_processor:
            return self.original_processor.summarize_text(text, max_len, min_len)
        
        # Last resort: simple truncation
        return text[:500] if len(text) > 500 else text
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using best available method.
        Returns enhanced format when available, standard format otherwise.
        """
        if self.use_enhanced and self.enhanced_processor:
            try:
                result = self.enhanced_processor.enhanced_sentiment_analysis(text)
                
                # Convert enhanced format to standard format for backward compatibility
                return {
                    "sentiment": result.get("sentiment", "neutral"),
                    "confidence": result.get("confidence", 0.5),
                    "probs": [
                        result.get("probabilities", {}).get("negative", 0.33),
                        result.get("probabilities", {}).get("neutral", 0.33),
                        result.get("probabilities", {}).get("positive", 0.33)
                    ],
                    "logits": result.get("logits", [0, 0, 0]),
                    # Enhanced fields (optional)
                    "enhanced": True,
                    "strength": result.get("strength"),
                    "market_implications": result.get("market_implications", [])
                }
            except Exception as e:
                logging.warning(f"Enhanced sentiment analysis failed, using fallback: {e}")
        
        # Fallback to original processor
        if self.original_processor:
            result = self.original_processor.analyze_sentiment(text)
            result["enhanced"] = False
            return result
        
        # Last resort: neutral sentiment
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "probs": [0.33, 0.34, 0.33],
            "logits": [0, 0, 0],
            "enhanced": False,
            "error": "No processor available"
        }
    
    # Enhanced methods (only available with enhanced processor)
    def extract_named_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities (enhanced feature only)"""
        if self.use_enhanced and self.enhanced_processor:
            try:
                return self.enhanced_processor.extract_named_entities(text)
            except Exception as e:
                logging.error(f"NER extraction failed: {e}")
        
        return {
            "entities": [],
            "companies": [],
            "people": [],
            "locations": [],
            "error": "Enhanced processor not available"
        }
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> list:
        """Extract key phrases (enhanced feature only)"""
        if self.use_enhanced and self.enhanced_processor:
            try:
                return self.enhanced_processor.extract_key_phrases(text, max_phrases)
            except Exception as e:
                logging.error(f"Key phrase extraction failed: {e}")
        
        return []
    
    def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive analysis (enhanced feature only)"""
        if self.use_enhanced and self.enhanced_processor:
            try:
                return self.enhanced_processor.comprehensive_analysis(text)
            except Exception as e:
                logging.error(f"Comprehensive analysis failed: {e}")
                return {"error": str(e), "fallback": True}
        
        # Provide basic analysis using available methods
        return {
            "summary": {"summary": self.summarize_text(text)},
            "sentiment": self.analyze_sentiment(text),
            "entities": {"companies": [], "people": [], "locations": []},
            "key_phrases": [],
            "enhanced": False,
            "message": "Using basic analysis - enhanced features not available"
        }
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get available capabilities"""
        return {
            "basic_summarization": self.original_processor is not None,
            "basic_sentiment": self.original_processor is not None,
            "enhanced_summarization": self.use_enhanced,
            "enhanced_sentiment": self.use_enhanced,
            "named_entity_recognition": self.use_enhanced,
            "key_phrase_extraction": self.use_enhanced,
            "comprehensive_analysis": self.use_enhanced,
            "market_implications": self.use_enhanced
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status for debugging"""
        return {
            "enhanced_available": self.use_enhanced,
            "original_available": self.original_processor is not None,
            "active_processor": "enhanced" if self.use_enhanced else "original",
            "capabilities": self.get_capabilities()
        }