#!/usr/bin/env python3
"""
Vector Search and Semantic Analysis Service

Provides semantic search capabilities using ChromaDB and sentence transformers
for content discovery, article recommendations, and similarity analysis.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from ..database.models import Article, User
from ..utils.text_processing import clean_text, extract_keywords


@dataclass
class SearchResult:
    """Search result with similarity score"""
    article_id: int
    title: str
    content: str
    similarity_score: float
    published_at: datetime
    symbols: List[str]
    sectors: List[str]
    metadata: Dict[str, Any]


@dataclass
class RecommendationResult:
    """Content recommendation result"""
    article_id: int
    title: str
    content_snippet: str
    recommendation_score: float
    reason: str
    published_at: datetime
    symbols: List[str]


class VectorSearchEngine:
    """Vector-based search and recommendation engine"""
    
    def __init__(self, persist_directory: str = "data/chromadb", 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector search engine
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            model_name: Sentence transformer model name
        """
        self.logger = logging.getLogger(__name__)
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Initialize components
        self.client = None
        self.collection = None
        self.model = None
        self.fallback_mode = False
        
        # Initialize ChromaDB and sentence transformers
        self._initialize_chromadb()
        self._initialize_sentence_transformer()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        if not CHROMADB_AVAILABLE:
            self.logger.warning("ChromaDB not available, using fallback mode")
            self.fallback_mode = True
            return
        
        try:
            # Ensure persist directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="finbrief_articles",
                metadata={"description": "FinBrief financial news articles"}
            )
            
            self.logger.info(f"ChromaDB initialized with {self.collection.count()} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            self.fallback_mode = True
    
    def _initialize_sentence_transformer(self):
        """Initialize sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("SentenceTransformers not available, using fallback embeddings")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Sentence transformer model loaded: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer: {e}")
            self.model = None
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding"""
        if self.model:
            try:
                embedding = self.model.encode([text])[0]
                return embedding.tolist()
            except Exception as e:
                self.logger.error(f"Failed to generate embedding: {e}")
        
        # Fallback: simple TF-IDF-like embedding
        return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Generate fallback embedding using simple hashing"""
        words = clean_text(text).split()
        
        # Simple hash-based embedding
        embedding = [0.0] * dim
        for word in words:
            hash_val = hash(word) % dim
            embedding[hash_val] += 1.0
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding
    
    async def add_article(self, article: Dict[str, Any]) -> bool:
        """Add article to vector search index"""
        try:
            article_id = str(article['id'])
            
            # Prepare text for embedding
            content = f"{article['title']} {article.get('content', '')}"
            content = clean_text(content)
            
            if self.fallback_mode or not self.collection:
                # Store in fallback storage (would implement file-based storage)
                return await self._add_article_fallback(article, content)
            
            # Generate embedding
            embedding = self._generate_embedding(content)
            
            # Prepare metadata
            metadata = {
                "title": article['title'],
                "published_at": article['published_at'].isoformat() if isinstance(article['published_at'], datetime) else str(article['published_at']),
                "symbols": json.dumps(article.get('symbols', [])),
                "sectors": json.dumps(article.get('sectors', [])),
                "url": article.get('url', ''),
                "source": article.get('source', ''),
                "sentiment": article.get('sentiment', 0.0),
                "importance": article.get('importance', 0.0)
            }
            
            # Add to ChromaDB
            self.collection.add(
                ids=[article_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )
            
            self.logger.debug(f"Added article {article_id} to vector index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add article to vector index: {e}")
            return False
    
    async def _add_article_fallback(self, article: Dict[str, Any], content: str) -> bool:
        """Fallback method to store article without ChromaDB"""
        # Would implement simple file-based storage
        return True
    
    async def search_similar(self, query: str, limit: int = 10, 
                           filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search for articles similar to query"""
        try:
            if self.fallback_mode or not self.collection:
                return await self._search_fallback(query, limit, filters)
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                if 'symbols' in filters and filters['symbols']:
                    # ChromaDB doesn't support array contains, so we'll filter post-search
                    pass
                if 'date_from' in filters:
                    where_clause['published_at'] = {'$gte': filters['date_from'].isoformat()}
                if 'min_sentiment' in filters:
                    where_clause['sentiment'] = {'$gte': filters['min_sentiment']}
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Get more results for post-filtering
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score (1 - normalized_distance)
                similarity_score = max(0.0, 1.0 - distance)
                
                # Apply symbol filtering if needed
                if filters and 'symbols' in filters and filters['symbols']:
                    article_symbols = json.loads(metadata.get('symbols', '[]'))
                    if not any(symbol in article_symbols for symbol in filters['symbols']):
                        continue
                
                # Parse metadata
                symbols = json.loads(metadata.get('symbols', '[]'))
                sectors = json.loads(metadata.get('sectors', '[]'))
                published_at = datetime.fromisoformat(metadata['published_at'])
                
                search_result = SearchResult(
                    article_id=int(results['ids'][0][i]),
                    title=metadata['title'],
                    content=doc[:500] + '...' if len(doc) > 500 else doc,
                    similarity_score=similarity_score,
                    published_at=published_at,
                    symbols=symbols,
                    sectors=sectors,
                    metadata=metadata
                )
                
                search_results.append(search_result)
                
                if len(search_results) >= limit:
                    break
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to search similar articles: {e}")
            return []
    
    async def _search_fallback(self, query: str, limit: int, 
                             filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Fallback search using keyword matching"""
        # Simple keyword-based search as fallback
        keywords = extract_keywords(query)
        
        # This would integrate with your database to find articles
        # For now, return empty list
        return []
    
    async def get_recommendations(self, user_id: int, article_ids: List[int] = None, 
                                limit: int = 10) -> List[RecommendationResult]:
        """Get content recommendations for user"""
        try:
            if self.fallback_mode or not self.collection:
                return await self._get_recommendations_fallback(user_id, article_ids, limit)
            
            # Get user reading history/preferences
            user_profile = await self._build_user_profile(user_id, article_ids)
            
            if not user_profile:
                # No user data, return popular articles
                return await self._get_popular_articles(limit)
            
            # Generate user profile embedding
            profile_embedding = self._generate_embedding(user_profile['content'])
            
            # Search for similar content
            results = self.collection.query(
                query_embeddings=[profile_embedding],
                n_results=limit * 2,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process recommendations
            recommendations = []
            seen_articles = set(article_ids or [])
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                article_id = int(results['ids'][0][i])
                
                # Skip articles user has already seen
                if article_id in seen_articles:
                    continue
                
                similarity_score = max(0.0, 1.0 - distance)
                
                # Calculate recommendation score based on multiple factors
                recency_score = self._calculate_recency_score(metadata['published_at'])
                importance_score = float(metadata.get('importance', 0.5))
                
                recommendation_score = (
                    similarity_score * 0.6 +
                    recency_score * 0.2 +
                    importance_score * 0.2
                )
                
                symbols = json.loads(metadata.get('symbols', '[]'))
                
                recommendation = RecommendationResult(
                    article_id=article_id,
                    title=metadata['title'],
                    content_snippet=doc[:200] + '...' if len(doc) > 200 else doc,
                    recommendation_score=recommendation_score,
                    reason=f"Based on your interest in {user_profile['interests'][:2]}",
                    published_at=datetime.fromisoformat(metadata['published_at']),
                    symbols=symbols
                )
                
                recommendations.append(recommendation)
                seen_articles.add(article_id)
                
                if len(recommendations) >= limit:
                    break
            
            # Sort by recommendation score
            recommendations.sort(key=lambda x: x.recommendation_score, reverse=True)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            return []
    
    async def _build_user_profile(self, user_id: int, 
                                article_ids: List[int] = None) -> Dict[str, Any]:
        """Build user interest profile from reading history"""
        try:
            # This would query your database for user's reading history
            # For now, return a mock profile
            
            if article_ids:
                # Use provided articles to build profile
                articles_content = []
                interests = []
                
                # Get article content from ChromaDB
                for article_id in article_ids[-10:]:  # Use recent 10 articles
                    try:
                        results = self.collection.get(
                            ids=[str(article_id)],
                            include=['documents', 'metadatas']
                        )
                        
                        if results['documents']:
                            articles_content.append(results['documents'][0])
                            metadata = results['metadatas'][0]
                            symbols = json.loads(metadata.get('symbols', '[]'))
                            sectors = json.loads(metadata.get('sectors', '[]'))
                            interests.extend(symbols + sectors)
                    except:
                        continue
                
                combined_content = ' '.join(articles_content)
                unique_interests = list(set(interests))
                
                return {
                    'content': combined_content,
                    'interests': unique_interests,
                    'user_id': user_id
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to build user profile: {e}")
            return None
    
    def _calculate_recency_score(self, published_at_str: str) -> float:
        """Calculate recency score (1.0 = very recent, 0.0 = very old)"""
        try:
            published_at = datetime.fromisoformat(published_at_str)
            hours_old = (datetime.now() - published_at).total_seconds() / 3600
            
            # Score decays exponentially over 7 days
            recency_score = max(0.0, np.exp(-hours_old / 168))  # 168 hours = 7 days
            
            return recency_score
            
        except:
            return 0.5
    
    async def _get_recommendations_fallback(self, user_id: int, 
                                          article_ids: List[int], 
                                          limit: int) -> List[RecommendationResult]:
        """Fallback recommendations without vector search"""
        # Would implement rule-based recommendations
        return []
    
    async def _get_popular_articles(self, limit: int) -> List[RecommendationResult]:
        """Get popular articles as default recommendations"""
        try:
            if not self.collection:
                return []
            
            # Get recent articles with high importance
            results = self.collection.query(
                query_embeddings=None,
                n_results=limit,
                where={'importance': {'$gte': 0.7}},
                include=['documents', 'metadatas']
            )
            
            recommendations = []
            for i, (doc, metadata) in enumerate(zip(
                results['documents'][0] if results['documents'] else [],
                results['metadatas'][0] if results['metadatas'] else []
            )):
                article_id = int(results['ids'][0][i])
                
                recommendation = RecommendationResult(
                    article_id=article_id,
                    title=metadata['title'],
                    content_snippet=doc[:200] + '...' if len(doc) > 200 else doc,
                    recommendation_score=float(metadata.get('importance', 0.5)),
                    reason="Popular article",
                    published_at=datetime.fromisoformat(metadata['published_at']),
                    symbols=json.loads(metadata.get('symbols', '[]'))
                )
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get popular articles: {e}")
            return []
    
    async def find_similar_articles(self, article_id: int, 
                                  limit: int = 5) -> List[SearchResult]:
        """Find articles similar to a given article"""
        try:
            if self.fallback_mode or not self.collection:
                return []
            
            # Get the article
            results = self.collection.get(
                ids=[str(article_id)],
                include=['documents', 'embeddings']
            )
            
            if not results['documents']:
                return []
            
            article_content = results['documents'][0]
            
            # Use the article content to find similar ones
            return await self.search_similar(
                query=article_content,
                limit=limit,
                filters={'exclude_id': article_id}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to find similar articles: {e}")
            return []
    
    async def get_trending_topics(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get trending topics based on vector clustering"""
        try:
            if self.fallback_mode or not self.collection:
                return []
            
            # Get recent articles
            from_date = datetime.now() - timedelta(days=days)
            
            results = self.collection.query(
                query_embeddings=None,
                n_results=1000,  # Get many articles
                where={'published_at': {'$gte': from_date.isoformat()}},
                include=['documents', 'metadatas']
            )
            
            if not results['documents']:
                return []
            
            # Analyze trending topics (simplified implementation)
            topic_counts = {}
            symbol_counts = {}
            
            for metadata in results['metadatas'][0]:
                symbols = json.loads(metadata.get('symbols', '[]'))
                sectors = json.loads(metadata.get('sectors', '[]'))
                
                for symbol in symbols:
                    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                
                for sector in sectors:
                    topic_counts[sector] = topic_counts.get(sector, 0) + 1
            
            # Get top trending topics
            trending_topics = []
            
            # Top symbols
            top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for symbol, count in top_symbols:
                trending_topics.append({
                    'topic': symbol,
                    'type': 'symbol',
                    'frequency': count,
                    'trend_score': count / len(results['documents'][0])
                })
            
            # Top sectors
            top_sectors = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for sector, count in top_sectors:
                trending_topics.append({
                    'topic': sector,
                    'type': 'sector', 
                    'frequency': count,
                    'trend_score': count / len(results['documents'][0])
                })
            
            return trending_topics
            
        except Exception as e:
            self.logger.error(f"Failed to get trending topics: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get vector index statistics"""
        try:
            stats = {
                'chromadb_available': CHROMADB_AVAILABLE,
                'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
                'fallback_mode': self.fallback_mode,
                'model_name': self.model_name,
                'total_documents': 0,
                'index_size': 0
            }
            
            if self.collection:
                stats['total_documents'] = self.collection.count()
                # Would add more detailed stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {'error': str(e)}


# Singleton instance
vector_search_engine = VectorSearchEngine()


# Utility functions for easy access

async def add_article_to_index(article: Dict[str, Any]) -> bool:
    """Add article to vector search index"""
    return await vector_search_engine.add_article(article)


async def search_articles(query: str, limit: int = 10, 
                         filters: Dict[str, Any] = None) -> List[SearchResult]:
    """Search articles by semantic similarity"""
    return await vector_search_engine.search_similar(query, limit, filters)


async def get_article_recommendations(user_id: int, article_ids: List[int] = None,
                                    limit: int = 10) -> List[RecommendationResult]:
    """Get personalized article recommendations"""
    return await vector_search_engine.get_recommendations(user_id, article_ids, limit)


async def find_related_articles(article_id: int, limit: int = 5) -> List[SearchResult]:
    """Find articles related to a given article"""
    return await vector_search_engine.find_similar_articles(article_id, limit)


async def get_trending_topics(days: int = 7) -> List[Dict[str, Any]]:
    """Get trending topics and symbols"""
    return await vector_search_engine.get_trending_topics(days)