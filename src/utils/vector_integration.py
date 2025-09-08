#!/usr/bin/env python3
"""
Vector Search Integration Utilities

Utilities to integrate vector search capabilities with the existing
news processing pipeline and database operations.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..services.vector_search import add_article_to_index, vector_search_engine
from ..database.models import Article, User
from ..database.database import get_db


class VectorSearchIntegrator:
    """Integrates vector search with existing pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.batch_size = 50
        self.processing_queue = []
    
    async def process_new_article(self, article_data: Dict[str, Any]) -> bool:
        """
        Process a newly crawled article for vector indexing
        
        Args:
            article_data: Article data from the crawling pipeline
            
        Returns:
            bool: True if successfully added to index
        """
        try:
            # Prepare article for vector indexing
            vector_article = self._prepare_article_for_indexing(article_data)
            
            # Add to vector search index
            success = await add_article_to_index(vector_article)
            
            if success:
                self.logger.info(f"Added article {article_data.get('id')} to vector index")
            else:
                self.logger.warning(f"Failed to add article {article_data.get('id')} to vector index")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to process article for vector search: {e}")
            return False
    
    def _prepare_article_for_indexing(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare article data for vector indexing"""
        return {
            'id': article_data.get('id'),
            'title': article_data.get('title', ''),
            'content': article_data.get('content', ''),
            'published_at': article_data.get('published_at', datetime.now()),
            'symbols': article_data.get('symbols', []),
            'sectors': article_data.get('sectors', []),
            'url': article_data.get('url', ''),
            'source': article_data.get('source', ''),
            'sentiment': article_data.get('sentiment', 0.0),
            'importance': article_data.get('importance', 0.5)
        }
    
    async def batch_index_articles(self, article_ids: List[int] = None, 
                                 limit: int = 1000) -> Dict[str, Any]:
        """
        Batch index articles from database
        
        Args:
            article_ids: Specific article IDs to index (optional)
            limit: Maximum number of articles to index
            
        Returns:
            Dict with indexing statistics
        """
        try:
            stats = {
                'total_processed': 0,
                'successful_indexes': 0,
                'failed_indexes': 0,
                'start_time': datetime.now(),
                'articles_per_second': 0.0
            }
            
            # Get database session
            db = next(get_db())
            
            try:
                # Query articles to index
                query = db.query(Article)
                
                if article_ids:
                    query = query.filter(Article.id.in_(article_ids))
                else:
                    # Get recent articles if no specific IDs provided
                    query = query.order_by(desc(Article.published_at))
                
                articles = query.limit(limit).all()
                
                self.logger.info(f"Starting batch indexing of {len(articles)} articles")
                
                # Process articles in batches
                for i in range(0, len(articles), self.batch_size):
                    batch = articles[i:i + self.batch_size]
                    await self._process_article_batch(batch, stats)
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.1)
                
                # Calculate final stats
                elapsed_time = (datetime.now() - stats['start_time']).total_seconds()
                if elapsed_time > 0:
                    stats['articles_per_second'] = stats['total_processed'] / elapsed_time
                
                stats['end_time'] = datetime.now()
                stats['total_time_seconds'] = elapsed_time
                
                self.logger.info(f"Batch indexing completed: {stats}")
                
                return stats
                
            finally:
                db.close()
                
        except Exception as e:
            self.logger.error(f"Batch indexing failed: {e}")
            return {'error': str(e)}
    
    async def _process_article_batch(self, articles: List[Article], 
                                   stats: Dict[str, Any]) -> None:
        """Process a batch of articles for indexing"""
        tasks = []
        
        for article in articles:
            # Convert SQLAlchemy model to dict
            article_data = {
                'id': article.id,
                'title': article.title,
                'content': article.content,
                'published_at': article.published_at,
                'symbols': article.symbols or [],
                'sectors': article.sectors or [],
                'url': article.url,
                'source': article.source,
                'sentiment': getattr(article, 'sentiment', 0.0),
                'importance': getattr(article, 'importance', 0.5)
            }
            
            # Create indexing task
            task = add_article_to_index(article_data)
            tasks.append((article.id, task))
        
        # Execute all tasks concurrently
        for article_id, task in tasks:
            try:
                success = await task
                stats['total_processed'] += 1
                
                if success:
                    stats['successful_indexes'] += 1
                else:
                    stats['failed_indexes'] += 1
                    self.logger.warning(f"Failed to index article {article_id}")
                    
            except Exception as e:
                stats['total_processed'] += 1
                stats['failed_indexes'] += 1
                self.logger.error(f"Error indexing article {article_id}: {e}")
    
    async def update_article_index(self, article_id: int, 
                                 updated_data: Dict[str, Any]) -> bool:
        """
        Update an existing article in the vector index
        
        Args:
            article_id: ID of article to update
            updated_data: Updated article data
            
        Returns:
            bool: True if successfully updated
        """
        try:
            # For now, we'll re-add the article (ChromaDB handles updates by re-adding)
            vector_article = self._prepare_article_for_indexing(updated_data)
            success = await add_article_to_index(vector_article)
            
            if success:
                self.logger.info(f"Updated article {article_id} in vector index")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update article {article_id} in vector index: {e}")
            return False
    
    def remove_article_from_index(self, article_id: int) -> bool:
        """
        Remove article from vector index
        
        Args:
            article_id: ID of article to remove
            
        Returns:
            bool: True if successfully removed
        """
        try:
            if vector_search_engine.collection:
                vector_search_engine.collection.delete(ids=[str(article_id)])
                self.logger.info(f"Removed article {article_id} from vector index")
                return True
            else:
                self.logger.warning("Vector search collection not available for deletion")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove article {article_id} from vector index: {e}")
            return False
    
    async def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        try:
            # Get vector search stats
            vector_stats = vector_search_engine.get_index_stats()
            
            # Get database article count
            db = next(get_db())
            try:
                total_articles = db.query(Article).count()
                recent_articles = db.query(Article).filter(
                    Article.published_at >= datetime.now().replace(hour=0, minute=0, second=0)
                ).count()
            finally:
                db.close()
            
            integration_stats = {
                'vector_index': vector_stats,
                'database_articles': total_articles,
                'todays_articles': recent_articles,
                'indexing_coverage': vector_stats.get('total_documents', 0) / max(total_articles, 1),
                'integration_status': 'healthy' if not vector_stats.get('fallback_mode') else 'degraded',
                'last_checked': datetime.now().isoformat()
            }
            
            return integration_stats
            
        except Exception as e:
            self.logger.error(f"Failed to get integration stats: {e}")
            return {'error': str(e)}


# Singleton integrator instance
vector_integrator = VectorSearchIntegrator()


# Pipeline integration hooks

async def on_article_processed(article_data: Dict[str, Any]) -> None:
    """
    Hook to be called when an article is processed by the pipeline
    
    Args:
        article_data: Article data from the pipeline
    """
    try:
        await vector_integrator.process_new_article(article_data)
    except Exception as e:
        logging.getLogger(__name__).error(f"Vector search integration hook failed: {e}")


async def on_article_updated(article_id: int, updated_data: Dict[str, Any]) -> None:
    """
    Hook to be called when an article is updated
    
    Args:
        article_id: ID of the updated article
        updated_data: Updated article data
    """
    try:
        await vector_integrator.update_article_index(article_id, updated_data)
    except Exception as e:
        logging.getLogger(__name__).error(f"Vector search update hook failed: {e}")


def on_article_deleted(article_id: int) -> None:
    """
    Hook to be called when an article is deleted
    
    Args:
        article_id: ID of the deleted article
    """
    try:
        vector_integrator.remove_article_from_index(article_id)
    except Exception as e:
        logging.getLogger(__name__).error(f"Vector search deletion hook failed: {e}")


# Utility functions

async def initialize_vector_search() -> bool:
    """
    Initialize vector search system and perform initial indexing
    
    Returns:
        bool: True if initialization successful
    """
    try:
        logging.getLogger(__name__).info("Initializing vector search system...")
        
        # Check if index is empty and needs initial population
        stats = vector_search_engine.get_index_stats()
        
        if stats.get('total_documents', 0) == 0:
            logging.getLogger(__name__).info("Vector index is empty, performing initial indexing...")
            
            # Index recent articles (last 30 days)
            result = await vector_integrator.batch_index_articles(limit=1000)
            
            if 'error' not in result:
                logging.getLogger(__name__).info(f"Initial indexing completed: {result['successful_indexes']} articles indexed")
                return True
            else:
                logging.getLogger(__name__).error(f"Initial indexing failed: {result['error']}")
                return False
        else:
            logging.getLogger(__name__).info(f"Vector index already contains {stats['total_documents']} documents")
            return True
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Vector search initialization failed: {e}")
        return False


async def reindex_all_articles() -> Dict[str, Any]:
    """
    Reindex all articles in the database
    
    Returns:
        Dict: Indexing statistics
    """
    try:
        logging.getLogger(__name__).info("Starting full reindexing of all articles...")
        return await vector_integrator.batch_index_articles(limit=10000)  # Large limit for full reindex
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Full reindexing failed: {e}")
        return {'error': str(e)}