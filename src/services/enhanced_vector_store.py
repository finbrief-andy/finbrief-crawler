"""
Enhanced Vector Store Service with advanced semantic search and recommendations.
Extends ChromaDB with multiple embedding models, semantic clustering, and recommendation engines.
"""
import os
import json
import logging
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer, util
    import sklearn.cluster
    from sklearn.metrics.pairwise import cosine_similarity
    CHROMA_AVAILABLE = True
except ImportError as e:
    CHROMA_AVAILABLE = False
    missing_deps = str(e).split("'")[1] if "'" in str(e) else "dependencies"
    logging.warning(f"Missing {missing_deps}. Vector storage disabled. Install: pip install chromadb sentence-transformers scikit-learn")

from sqlalchemy.orm import Session
from src.database.models_migration import News, Analysis, MarketEnum, AssetTypeEnum


class EnhancedVectorStore:
    """Enhanced vector storage with advanced semantic search and recommendations"""
    
    def __init__(self, persist_directory: str = "./data/vectors", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.logger = logging.getLogger(__name__)
        
        # Model performance tracking
        self.model_metrics = {
            "embeddings_generated": 0,
            "searches_performed": 0,
            "recommendations_served": 0,
            "cache_hits": 0
        }
        
        if not CHROMA_AVAILABLE:
            self.logger.error("ChromaDB/sentence-transformers not available. Install dependencies first.")
            self.client = None
            self.embedding_model = None
            self.financial_model = None
            return
        
        # Initialize components
        try:
            self._initialize_chromadb()
            self._initialize_embedding_models()
            self._initialize_collections()
            self._setup_caching()
            
            self.logger.info(f"Enhanced vector store initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced vector store: {e}")
            self.client = None
            self.embedding_model = None
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB with optimized settings"""
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Use optimized settings for production
        settings = Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False,  # Disable telemetry for privacy
            allow_reset=False  # Prevent accidental data loss
        )
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=settings
        )
    
    def _initialize_embedding_models(self):
        """Initialize multiple embedding models for different use cases"""
        # General purpose embedding model (fast, good quality)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Financial-specific model (if available)
        try:
            # Use a financial domain-specific model if available
            self.financial_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.logger.info("✅ Financial embedding model loaded")
        except Exception as e:
            self.financial_model = self.embedding_model  # Fallback
            self.logger.info("Using general model for financial embeddings")
        
        # Model warmup
        self._warmup_models()
    
    def _warmup_models(self):
        """Warm up embedding models for better performance"""
        try:
            warmup_texts = [
                "Stock market analysis",
                "Financial earnings report", 
                "Investment recommendation"
            ]
            self.embedding_model.encode(warmup_texts)
            self.financial_model.encode(warmup_texts)
            self.logger.info("✅ Embedding models warmed up")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections with enhanced metadata"""
        # News articles collection
        self.news_collection = self.client.get_or_create_collection(
            name="news_embeddings_v2",
            metadata={
                "description": "Enhanced news article embeddings with financial context",
                "embedding_model": self.embedding_model_name,
                "version": "2.0"
            }
        )
        
        # Analysis results collection
        self.analysis_collection = self.client.get_or_create_collection(
            name="analysis_embeddings_v2",
            metadata={
                "description": "Analysis result embeddings with sentiment context",
                "embedding_model": self.embedding_model_name,
                "version": "2.0"
            }
        )
        
        # Strategy insights collection (new)
        self.strategy_collection = self.client.get_or_create_collection(
            name="strategy_embeddings",
            metadata={
                "description": "Investment strategy embeddings for recommendation",
                "embedding_model": self.embedding_model_name
            }
        )
        
        # Entity mentions collection (new)
        self.entity_collection = self.client.get_or_create_collection(
            name="entity_mentions",
            metadata={
                "description": "Company and entity mention embeddings",
                "embedding_model": self.embedding_model_name
            }
        )
        
        # Log collection sizes
        news_count = len(self.news_collection.get()['ids'])
        analysis_count = len(self.analysis_collection.get()['ids'])
        
        self.logger.info(f"Collections initialized: {news_count} news, {analysis_count} analyses")
    
    def _setup_caching(self):
        """Setup embedding cache for performance"""
        self.embedding_cache = {}
        self.cache_max_size = 1000  # Limit cache size
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def is_available(self) -> bool:
        """Check if vector store is available"""
        return (self.client is not None and 
                self.embedding_model is not None and 
                CHROMA_AVAILABLE)
    
    def generate_embeddings(self, texts: List[str], use_financial_model: bool = False, 
                          use_cache: bool = True) -> List[List[float]]:
        """Generate embeddings with caching and model selection"""
        if not self.is_available():
            return []
        
        try:
            embeddings = []
            cache_misses = []
            cache_miss_indices = []
            
            # Check cache first
            for i, text in enumerate(texts):
                if use_cache:
                    cache_key = self._get_cache_key(text)
                    if cache_key in self.embedding_cache:
                        embeddings.append(self.embedding_cache[cache_key])
                        self.model_metrics["cache_hits"] += 1
                    else:
                        cache_misses.append(text)
                        cache_miss_indices.append(i)
                        embeddings.append(None)  # Placeholder
                else:
                    cache_misses.append(text)
                    cache_miss_indices.append(i)
                    embeddings.append(None)
            
            # Generate embeddings for cache misses
            if cache_misses:
                model = self.financial_model if use_financial_model else self.embedding_model
                new_embeddings = model.encode(cache_misses, convert_to_numpy=True)
                
                # Insert new embeddings and update cache
                for i, embedding in enumerate(new_embeddings):
                    original_index = cache_miss_indices[i]
                    embeddings[original_index] = embedding.tolist()
                    
                    # Update cache
                    if use_cache and len(self.embedding_cache) < self.cache_max_size:
                        cache_key = self._get_cache_key(cache_misses[i])
                        self.embedding_cache[cache_key] = embedding.tolist()
                
                self.model_metrics["embeddings_generated"] += len(cache_misses)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return []
    
    def add_news_embedding(self, news: News, enhanced_metadata: bool = True) -> bool:
        """Add news article with enhanced metadata"""
        if not self.is_available():
            return False
        
        try:
            # Create rich searchable text
            text_parts = [news.headline]
            if news.content_summary:
                text_parts.append(news.content_summary)
            if news.tickers:
                text_parts.append(f"Related companies: {', '.join(news.tickers[:5])}")
            
            text = ". ".join(text_parts)
            
            # Generate embedding using financial model for financial content
            embedding = self.generate_embeddings([text], use_financial_model=True)
            
            if not embedding:
                return False
            
            # Enhanced metadata
            metadata = {
                "source": news.source,
                "market": news.market.value,
                "asset_type": news.asset_type.value if news.asset_type else "stocks",
                "published_at": news.published_at.isoformat(),
                "tickers": json.dumps(news.tickers or []),
                "ticker_count": len(news.tickers or []),
                "headline": news.headline[:200],
                "url": news.url or "",
                "content_length": len(news.content_summary or ""),
                "hour_of_day": news.published_at.hour,
                "day_of_week": news.published_at.weekday(),
                "is_recent": (datetime.utcnow() - news.published_at).days < 7
            }
            
            # Add enhanced metadata for better filtering
            if enhanced_metadata:
                metadata.update({
                    "has_tickers": len(news.tickers or []) > 0,
                    "is_major_source": news.source.lower() in ["reuters", "bloomberg", "marketwatch", "cnbc"],
                    "word_count": len((news.content_summary or "").split()),
                    "urgency_score": self._calculate_urgency_score(news)
                })
            
            self.news_collection.add(
                embeddings=embedding,
                documents=[text],
                metadatas=[metadata],
                ids=[f"news_{news.id}"]
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding enhanced news embedding for id={news.id}: {e}")
            return False
    
    def _calculate_urgency_score(self, news: News) -> float:
        """Calculate urgency score based on content and timing"""
        score = 0.5  # Base score
        
        urgent_keywords = ["breaking", "urgent", "alert", "crash", "surge", "plunge", "bankruptcy"]
        headline_lower = news.headline.lower()
        
        for keyword in urgent_keywords:
            if keyword in headline_lower:
                score += 0.2
        
        # Recent news gets higher urgency
        hours_old = (datetime.utcnow() - news.published_at).total_seconds() / 3600
        if hours_old < 1:
            score += 0.3
        elif hours_old < 24:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def add_analysis_embedding(self, analysis: Analysis) -> bool:
        """Add analysis with enhanced context"""
        if not self.is_available() or not analysis.news:
            return False
        
        try:
            # Create rich analysis text
            text_parts = [analysis.news.headline]
            if analysis.rationale:
                text_parts.append(analysis.rationale)
            if analysis.sentiment:
                text_parts.append(f"Market sentiment: {analysis.sentiment.value}")
            
            # Add action context
            actions = []
            if analysis.action_short:
                actions.append(f"Short-term: {analysis.action_short.value}")
            if analysis.action_mid:
                actions.append(f"Mid-term: {analysis.action_mid.value}")
            if analysis.action_long:
                actions.append(f"Long-term: {analysis.action_long.value}")
            
            if actions:
                text_parts.append(". ".join(actions))
            
            text = ". ".join(text_parts)
            embedding = self.generate_embeddings([text], use_financial_model=True)
            
            if not embedding:
                return False
            
            # Enhanced analysis metadata
            metadata = {
                "news_id": analysis.news_id,
                "sentiment": analysis.sentiment.value if analysis.sentiment else "neutral",
                "sentiment_score": analysis.sentiment_score or 0.0,
                "confidence_score": analysis.confidence_score or 0.0,
                "action_short": analysis.action_short.value if analysis.action_short else "NONE",
                "action_mid": analysis.action_mid.value if analysis.action_mid else "NONE", 
                "action_long": analysis.action_long.value if analysis.action_long else "NONE",
                "created_at": analysis.created_at.isoformat(),
                "model_name": analysis.model_name or "unknown",
                "market": analysis.news.market.value,
                "has_rationale": bool(analysis.rationale),
                "rationale_length": len(analysis.rationale or ""),
                "action_consistency": self._calculate_action_consistency(analysis)
            }
            
            self.analysis_collection.add(
                embeddings=embedding,
                documents=[text],
                metadatas=[metadata],
                ids=[f"analysis_{analysis.id}"]
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding enhanced analysis embedding for id={analysis.id}: {e}")
            return False
    
    def _calculate_action_consistency(self, analysis: Analysis) -> float:
        """Calculate consistency score between short/mid/long term actions"""
        actions = [analysis.action_short, analysis.action_mid, analysis.action_long]
        valid_actions = [a for a in actions if a is not None]
        
        if len(valid_actions) <= 1:
            return 1.0
        
        # Simple consistency check - same direction actions score higher
        action_values = [a.value for a in valid_actions]
        buy_actions = sum(1 for a in action_values if "BUY" in a)
        sell_actions = sum(1 for a in action_values if "SELL" in a)
        hold_actions = sum(1 for a in action_values if "HOLD" in a)
        
        max_consistent = max(buy_actions, sell_actions, hold_actions)
        return max_consistent / len(valid_actions)
    
    def semantic_search(self, query: str, content_type: str = "news", limit: int = 10,
                       filters: Dict[str, Any] = None, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Advanced semantic search with filtering and similarity thresholds"""
        if not self.is_available():
            return []
        
        try:
            self.model_metrics["searches_performed"] += 1
            
            # Select appropriate collection
            if content_type == "news":
                collection = self.news_collection
            elif content_type == "analysis":
                collection = self.analysis_collection
            elif content_type == "strategy":
                collection = self.strategy_collection
            else:
                collection = self.news_collection  # Default
            
            # Generate query embedding
            query_embedding = self.generate_embeddings([query], use_financial_model=True)
            if not query_embedding:
                return []
            
            # Build ChromaDB where filter
            where_filter = {}
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        where_filter[key] = value
            
            # Perform search
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(limit * 2, 100),  # Get more results to filter by similarity
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process and filter results
            similar_items = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    # Convert distance to similarity (ChromaDB uses cosine distance)
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    # Apply similarity threshold
                    if similarity >= similarity_threshold:
                        similar_items.append({
                            "id": results["ids"][0][i],
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "similarity": similarity,
                            "distance": distance,
                            "relevance_score": self._calculate_relevance_score(
                                results["metadatas"][0][i], similarity
                            )
                        })
            
            # Sort by relevance score and limit results
            similar_items.sort(key=lambda x: x["relevance_score"], reverse=True)
            return similar_items[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def _calculate_relevance_score(self, metadata: Dict[str, Any], similarity: float) -> float:
        """Calculate relevance score combining similarity with metadata signals"""
        base_score = similarity
        
        # Boost recent content
        if metadata.get("is_recent", False):
            base_score += 0.1
        
        # Boost high-confidence analyses
        confidence = metadata.get("confidence_score", 0.5)
        if confidence > 0.8:
            base_score += 0.1
        
        # Boost major sources
        if metadata.get("is_major_source", False):
            base_score += 0.05
        
        # Boost content with tickers (more actionable)
        if metadata.get("has_tickers", False):
            base_score += 0.05
        
        return min(base_score, 1.0)
    
    def find_related_articles(self, news_id: int, limit: int = 5, 
                            market_filter: str = None) -> List[Dict[str, Any]]:
        """Find articles related to a specific news item"""
        try:
            # Get the original article embedding
            results = self.news_collection.get(ids=[f"news_{news_id}"])
            
            if not results["documents"]:
                return []
            
            original_text = results["documents"][0]
            
            # Search for similar articles, excluding the original
            filters = {}
            if market_filter:
                filters["market"] = market_filter
            
            similar = self.semantic_search(
                query=original_text,
                content_type="news", 
                limit=limit + 1,  # +1 to account for excluding original
                filters=filters
            )
            
            # Filter out the original article
            related = [item for item in similar if item["id"] != f"news_{news_id}"]
            
            return related[:limit]
            
        except Exception as e:
            self.logger.error(f"Error finding related articles: {e}")
            return []
    
    def generate_recommendations(self, user_interests: List[str], market: str = None,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on user interests"""
        if not self.is_available():
            return []
        
        try:
            self.model_metrics["recommendations_served"] += 1
            
            # Combine user interests into a query
            interest_query = ". ".join(user_interests)
            
            # Search for relevant content
            filters = {}
            if market:
                filters["market"] = market
            
            # Get news recommendations
            news_recs = self.semantic_search(
                query=interest_query,
                content_type="news",
                limit=limit // 2,
                filters=filters,
                similarity_threshold=0.4  # Slightly higher threshold for recommendations
            )
            
            # Get analysis recommendations
            analysis_recs = self.semantic_search(
                query=interest_query,
                content_type="analysis", 
                limit=limit // 2,
                filters=filters,
                similarity_threshold=0.4
            )
            
            # Combine and diversify recommendations
            all_recs = news_recs + analysis_recs
            all_recs.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Diversify by ensuring variety in sources and topics
            diverse_recs = self._diversify_recommendations(all_recs, limit)
            
            return diverse_recs
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _diversify_recommendations(self, recommendations: List[Dict[str, Any]], 
                                 limit: int) -> List[Dict[str, Any]]:
        """Diversify recommendations to avoid too much similar content"""
        if len(recommendations) <= limit:
            return recommendations
        
        diverse_recs = []
        seen_sources = set()
        seen_tickers = set()
        
        # First pass: ensure source diversity
        for rec in recommendations:
            if len(diverse_recs) >= limit:
                break
            
            source = rec["metadata"].get("source", "unknown")
            tickers_str = rec["metadata"].get("tickers", "[]")
            
            try:
                tickers = json.loads(tickers_str) if isinstance(tickers_str, str) else tickers_str
            except:
                tickers = []
            
            # Check diversity constraints
            source_ok = len(seen_sources) < 3 or source not in seen_sources
            ticker_diversity_ok = len(seen_tickers) < 5 or not any(t in seen_tickers for t in tickers[:2])
            
            if source_ok or ticker_diversity_ok:
                diverse_recs.append(rec)
                seen_sources.add(source)
                for ticker in tickers[:2]:  # Only consider first 2 tickers
                    seen_tickers.add(ticker)
        
        # Fill remaining slots if needed
        remaining_slots = limit - len(diverse_recs)
        if remaining_slots > 0:
            for rec in recommendations:
                if rec not in diverse_recs:
                    diverse_recs.append(rec)
                    remaining_slots -= 1
                    if remaining_slots == 0:
                        break
        
        return diverse_recs
    
    def cluster_similar_content(self, content_type: str = "news", n_clusters: int = 10) -> Dict[str, Any]:
        """Cluster similar content for topic discovery"""
        if not self.is_available():
            return {}
        
        try:
            # Get collection
            collection = self.news_collection if content_type == "news" else self.analysis_collection
            
            # Get all embeddings
            all_data = collection.get(include=["embeddings", "documents", "metadatas"])
            
            if not all_data["embeddings"]:
                return {}
            
            embeddings = np.array(all_data["embeddings"])
            
            # Perform clustering
            kmeans = sklearn.cluster.KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Organize results by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append({
                    "id": all_data["ids"][i],
                    "text": all_data["documents"][i][:200] + "...",
                    "metadata": all_data["metadatas"][i]
                })
            
            # Generate cluster summaries
            cluster_info = {}
            for cluster_id, items in clusters.items():
                cluster_info[cluster_id] = {
                    "size": len(items),
                    "items": items[:5],  # Show top 5 items per cluster
                    "common_topics": self._extract_cluster_topics(items)
                }
            
            return {
                "total_items": len(embeddings),
                "clusters": cluster_info,
                "n_clusters": len(clusters)
            }
            
        except Exception as e:
            self.logger.error(f"Error clustering content: {e}")
            return {}
    
    def _extract_cluster_topics(self, items: List[Dict[str, Any]]) -> List[str]:
        """Extract common topics from a cluster of items"""
        # Simple keyword extraction from headlines and metadata
        keywords = defaultdict(int)
        
        for item in items:
            # Extract from headline
            headline = item["metadata"].get("headline", "")
            words = headline.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():
                    keywords[word] += 1
            
            # Extract from tickers
            tickers_str = item["metadata"].get("tickers", "[]")
            try:
                tickers = json.loads(tickers_str) if isinstance(tickers_str, str) else []
                for ticker in tickers:
                    keywords[ticker.upper()] += 2  # Give tickers more weight
            except:
                pass
        
        # Return top keywords
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, count in top_keywords if count > 1]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store"""
        if not self.is_available():
            return {"error": "Vector store not available"}
        
        try:
            news_data = self.news_collection.get(include=["metadatas"])
            analysis_data = self.analysis_collection.get(include=["metadatas"])
            
            stats = {
                "collections": {
                    "news": {
                        "count": len(news_data["ids"]),
                        "sources": len(set(m.get("source", "unknown") for m in news_data["metadatas"])),
                        "markets": len(set(m.get("market", "unknown") for m in news_data["metadatas"]))
                    },
                    "analysis": {
                        "count": len(analysis_data["ids"]),
                        "sentiment_distribution": self._get_sentiment_distribution(analysis_data["metadatas"]),
                        "models": len(set(m.get("model_name", "unknown") for m in analysis_data["metadatas"]))
                    }
                },
                "performance_metrics": self.model_metrics.copy(),
                "cache_stats": {
                    "size": len(self.embedding_cache),
                    "max_size": self.cache_max_size,
                    "hit_rate": (self.model_metrics["cache_hits"] / 
                               max(1, self.model_metrics["embeddings_generated"] + self.model_metrics["cache_hits"]))
                },
                "models": {
                    "general": self.embedding_model_name,
                    "financial": "specialized" if self.financial_model != self.embedding_model else "general"
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def _get_sentiment_distribution(self, metadatas: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get sentiment distribution from analysis metadata"""
        distribution = defaultdict(int)
        for metadata in metadatas:
            sentiment = metadata.get("sentiment", "unknown")
            distribution[sentiment] += 1
        return dict(distribution)
    
    def cleanup_old_embeddings(self, days_old: int = 90):
        """Clean up old embeddings to manage storage"""
        if not self.is_available():
            return
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            cutoff_iso = cutoff_date.isoformat()
            
            # Get old news embeddings
            news_data = self.news_collection.get(include=["metadatas"])
            old_news_ids = []
            
            for i, metadata in enumerate(news_data["metadatas"]):
                published_at = metadata.get("published_at", "")
                if published_at and published_at < cutoff_iso:
                    old_news_ids.append(news_data["ids"][i])
            
            # Delete old embeddings
            if old_news_ids:
                self.news_collection.delete(ids=old_news_ids)
                self.logger.info(f"Cleaned up {len(old_news_ids)} old news embeddings")
            
            # Similar cleanup for analysis
            analysis_data = self.analysis_collection.get(include=["metadatas"])
            old_analysis_ids = []
            
            for i, metadata in enumerate(analysis_data["metadatas"]):
                created_at = metadata.get("created_at", "")
                if created_at and created_at < cutoff_iso:
                    old_analysis_ids.append(analysis_data["ids"][i])
            
            if old_analysis_ids:
                self.analysis_collection.delete(ids=old_analysis_ids)
                self.logger.info(f"Cleaned up {len(old_analysis_ids)} old analysis embeddings")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old embeddings: {e}")


# Global enhanced vector store instance
_enhanced_vector_store = None

def get_enhanced_vector_store() -> EnhancedVectorStore:
    """Get singleton enhanced vector store instance"""
    global _enhanced_vector_store
    if _enhanced_vector_store is None:
        _enhanced_vector_store = EnhancedVectorStore()
    return _enhanced_vector_store


# Backward compatibility - keep original interface working
def get_vector_store():
    """Backward compatible function - returns enhanced store"""
    return get_enhanced_vector_store()