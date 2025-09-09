"""
Vector Store Service for embeddings and semantic search.
Uses ChromaDB for local vector storage with sentence transformers for embeddings.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB or sentence-transformers not installed. Vector storage disabled.")

from sqlalchemy.orm import Session
from src.database.models_migration import News, Analysis


class VectorStore:
    """Vector storage and semantic search service"""
    
    def __init__(self, persist_directory: str = "./data/vectors"):
        self.persist_directory = persist_directory
        self.logger = logging.getLogger(__name__)
        
        if not CHROMA_AVAILABLE:
            self.logger.error("ChromaDB not available. Install with: pip install chromadb sentence-transformers")
            self.client = None
            self.embedding_model = None
            return
        
        # Initialize ChromaDB client
        try:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Initialize embedding model (lightweight multilingual model)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create collections for different content types
            self.news_collection = self.client.get_or_create_collection(
                name="news_embeddings",
                metadata={"description": "News article embeddings for semantic search"}
            )
            
            self.analysis_collection = self.client.get_or_create_collection(
                name="analysis_embeddings", 
                metadata={"description": "Analysis result embeddings"}
            )
            
            self.logger.info(f"Vector store initialized with {len(self.news_collection.get()['ids'])} news embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            self.client = None
            self.embedding_model = None
    
    def is_available(self) -> bool:
        """Check if vector store is available"""
        return self.client is not None and self.embedding_model is not None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts"""
        if not self.is_available():
            return []
        
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return []
    
    def add_news_embedding(self, news: News) -> bool:
        """Add news article to vector store"""
        if not self.is_available():
            return False
        
        try:
            # Create searchable text from news
            text = f"{news.headline}. {news.content_summary or ''}"
            embedding = self.generate_embeddings([text])
            
            if not embedding:
                return False
            
            # Metadata for filtering
            metadata = {
                "source": news.source,
                "market": news.market.value,
                "asset_type": news.asset_type.value if news.asset_type else "stocks",
                "published_at": news.published_at.isoformat(),
                "tickers": json.dumps(news.tickers or []),
                "headline": news.headline[:200],  # Truncate for storage
                "url": news.url or ""
            }
            
            self.news_collection.add(
                embeddings=embedding,
                documents=[text],
                metadatas=[metadata],
                ids=[f"news_{news.id}"]
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding news embedding for id={news.id}: {e}")
            return False
    
    def add_analysis_embedding(self, analysis: Analysis) -> bool:
        """Add analysis result to vector store"""
        if not self.is_available() or not analysis.news:
            return False
        
        try:
            # Create searchable text from analysis
            text = f"{analysis.news.headline}. {analysis.rationale or ''}"
            embedding = self.generate_embeddings([text])
            
            if not embedding:
                return False
            
            metadata = {
                "news_id": analysis.news_id,
                "sentiment": analysis.sentiment.value if analysis.sentiment else "neutral",
                "sentiment_score": analysis.sentiment_score or 0.0,
                "action_short": analysis.action_short.value if analysis.action_short else "NONE",
                "action_mid": analysis.action_mid.value if analysis.action_mid else "NONE", 
                "action_long": analysis.action_long.value if analysis.action_long else "NONE",
                "created_at": analysis.created_at.isoformat(),
                "model_name": analysis.model_name or "unknown"
            }
            
            self.analysis_collection.add(
                embeddings=embedding,
                documents=[text],
                metadatas=[metadata],
                ids=[f"analysis_{analysis.id}"]
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding analysis embedding for id={analysis.id}: {e}")
            return False
    
    def search_similar_news(self, query: str, limit: int = 10, 
                          market: str = None, asset_type: str = None) -> List[Dict[str, Any]]:
        """Search for similar news articles"""
        if not self.is_available():
            return []
        
        try:
            query_embedding = self.generate_embeddings([query])
            if not query_embedding:
                return []
            
            # Build filter conditions
            where_filter = {}
            if market:
                where_filter["market"] = market
            if asset_type:
                where_filter["asset_type"] = asset_type
            
            results = self.news_collection.query(
                query_embeddings=query_embedding,
                n_results=limit,
                where=where_filter if where_filter else None
            )
            
            # Format results
            similar_items = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    similar_items.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    })
            
            return similar_items
            
        except Exception as e:
            self.logger.error(f"Error searching similar news: {e}")
            return []
    
    def search_similar_analyses(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar analysis results"""
        if not self.is_available():
            return []
        
        try:
            query_embedding = self.generate_embeddings([query])
            if not query_embedding:
                return []
            
            results = self.analysis_collection.query(
                query_embeddings=query_embedding,
                n_results=limit
            )
            
            # Format results
            similar_items = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    similar_items.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    })
            
            return similar_items
            
        except Exception as e:
            self.logger.error(f"Error searching similar analyses: {e}")
            return []
    
    def get_relevant_context(self, query: str, limit: int = 5) -> str:
        """Get relevant context text for strategy generation"""
        similar_news = self.search_similar_news(query, limit=limit)
        
        if not similar_news:
            return ""
        
        context_parts = []
        for item in similar_news:
            headline = item["metadata"].get("headline", "")
            context_parts.append(f"• {headline}")
        
        return "\n".join(context_parts)
    
    def backfill_embeddings(self, session: Session, limit: int = 1000):
        """Backfill embeddings for existing news and analysis"""
        if not self.is_available():
            self.logger.warning("Vector store not available for backfill")
            return
        
        # Backfill news embeddings
        news_query = session.query(News).order_by(News.created_at.desc()).limit(limit)
        news_count = 0
        
        for news in news_query:
            if self.add_news_embedding(news):
                news_count += 1
        
        # Backfill analysis embeddings
        analysis_query = session.query(Analysis).join(News).order_by(Analysis.created_at.desc()).limit(limit)
        analysis_count = 0
        
        for analysis in analysis_query:
            if self.add_analysis_embedding(analysis):
                analysis_count += 1
        
        self.logger.info(f"Backfilled {news_count} news and {analysis_count} analysis embeddings")


# Global vector store instance
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """Get singleton vector store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance


if __name__ == "__main__":
    # Test vector store functionality
    vs = VectorStore()
    
    if vs.is_available():
        print("Vector store is available")
        
        # Test embedding generation
        texts = ["Gold prices rise on inflation fears", "Stock market volatility continues"]
        embeddings = vs.generate_embeddings(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Test search
        results = vs.search_similar_news("gold price increase")
        print(f"Found {len(results)} similar news items")
    else:
        print("Vector store not available - install dependencies: pip install chromadb sentence-transformers")