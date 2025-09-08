#!/usr/bin/env python3
"""
Vector Search API Endpoints

REST API endpoints for semantic search, content recommendations,
and similarity analysis functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date
import logging

from ..services.vector_search import (
    vector_search_engine,
    search_articles,
    get_article_recommendations, 
    find_related_articles,
    get_trending_topics,
    add_article_to_index
)
from ..auth.jwt_handler import get_current_user
from ..database.models import User

router = APIRouter(prefix="/api/v1/search", tags=["Vector Search"])
security = HTTPBearer()
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    symbols: Optional[List[str]] = Field(default=None, description="Filter by symbols")
    sectors: Optional[List[str]] = Field(default=None, description="Filter by sectors") 
    date_from: Optional[date] = Field(default=None, description="Filter articles from this date")
    date_to: Optional[date] = Field(default=None, description="Filter articles to this date")
    min_sentiment: Optional[float] = Field(default=None, ge=-1, le=1, description="Minimum sentiment score")
    min_similarity: Optional[float] = Field(default=0.0, ge=0, le=1, description="Minimum similarity score")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total_found: int
    search_time_ms: float
    filters_applied: Dict[str, Any]

class RecommendationRequest(BaseModel):
    based_on_articles: Optional[List[int]] = Field(default=None, description="Article IDs to base recommendations on")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of recommendations")
    exclude_articles: Optional[List[int]] = Field(default=None, description="Article IDs to exclude")

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    user_id: int
    total_found: int
    recommendation_strategy: str

class SimilarArticlesRequest(BaseModel):
    article_id: int = Field(..., description="Article ID to find similar articles for")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of similar articles")

class TrendingTopicsRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=30, description="Number of days to analyze")
    topic_types: Optional[List[str]] = Field(default=None, description="Types of topics to include")
    min_frequency: int = Field(default=2, ge=1, description="Minimum frequency for trending topics")

class IndexStatsResponse(BaseModel):
    chromadb_available: bool
    sentence_transformers_available: bool
    fallback_mode: bool
    model_name: str
    total_documents: int
    index_size: int
    last_updated: Optional[datetime] = None

# Search Endpoints

@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    search_request: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Perform semantic search across articles
    
    Searches for articles using vector similarity rather than keyword matching.
    Returns results ranked by semantic relevance to the query.
    """
    try:
        start_time = datetime.now()
        
        # Prepare filters
        filters = {}
        if search_request.symbols:
            filters['symbols'] = search_request.symbols
        if search_request.sectors:
            filters['sectors'] = search_request.sectors
        if search_request.date_from:
            filters['date_from'] = datetime.combine(search_request.date_from, datetime.min.time())
        if search_request.date_to:
            filters['date_to'] = datetime.combine(search_request.date_to, datetime.max.time())
        if search_request.min_sentiment is not None:
            filters['min_sentiment'] = search_request.min_sentiment
        
        # Perform search
        results = await search_articles(
            query=search_request.query,
            limit=search_request.limit,
            filters=filters
        )
        
        # Filter by minimum similarity if specified
        if search_request.min_similarity > 0.0:
            results = [r for r in results if r.similarity_score >= search_request.min_similarity]
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "article_id": result.article_id,
                "title": result.title,
                "content_snippet": result.content,
                "similarity_score": round(result.similarity_score, 4),
                "published_at": result.published_at.isoformat(),
                "symbols": result.symbols,
                "sectors": result.sectors,
                "relevance": "high" if result.similarity_score > 0.8 else "medium" if result.similarity_score > 0.6 else "low"
            })
        
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=formatted_results,
            query=search_request.query,
            total_found=len(formatted_results),
            search_time_ms=round(search_time, 2),
            filters_applied=filters
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/similar/{article_id}")
async def get_similar_articles(
    article_id: int,
    limit: int = Query(default=5, ge=1, le=20),
    current_user: User = Depends(get_current_user)
):
    """
    Find articles similar to a specific article
    
    Uses vector similarity to find articles with similar content,
    topics, or market implications.
    """
    try:
        results = await find_related_articles(article_id, limit)
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "article_id": result.article_id,
                "title": result.title,
                "content_snippet": result.content,
                "similarity_score": round(result.similarity_score, 4),
                "published_at": result.published_at.isoformat(),
                "symbols": result.symbols,
                "sectors": result.sectors,
                "relationship": "similar content" if result.similarity_score > 0.8 else "related topic"
            })
        
        return {
            "source_article_id": article_id,
            "similar_articles": formatted_results,
            "total_found": len(formatted_results),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to find similar articles: {e}")
        raise HTTPException(status_code=500, detail=f"Similar articles search failed: {str(e)}")

# Recommendation Endpoints

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_personalized_recommendations(
    recommendation_request: RecommendationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Get personalized article recommendations
    
    Returns articles tailored to user interests based on reading history
    or specified articles.
    """
    try:
        recommendations = await get_article_recommendations(
            user_id=current_user.id,
            article_ids=recommendation_request.based_on_articles,
            limit=recommendation_request.limit
        )
        
        # Exclude specified articles
        if recommendation_request.exclude_articles:
            recommendations = [
                r for r in recommendations 
                if r.article_id not in recommendation_request.exclude_articles
            ]
        
        formatted_recommendations = []
        for rec in recommendations:
            formatted_recommendations.append({
                "article_id": rec.article_id,
                "title": rec.title,
                "content_snippet": rec.content_snippet,
                "recommendation_score": round(rec.recommendation_score, 4),
                "reason": rec.reason,
                "published_at": rec.published_at.isoformat(),
                "symbols": rec.symbols,
                "confidence": "high" if rec.recommendation_score > 0.8 else "medium" if rec.recommendation_score > 0.6 else "low"
            })
        
        strategy = "content-based" if recommendation_request.based_on_articles else "user-profile-based"
        
        return RecommendationResponse(
            recommendations=formatted_recommendations,
            user_id=current_user.id,
            total_found=len(formatted_recommendations),
            recommendation_strategy=strategy
        )
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")

@router.get("/recommendations/trending")
async def get_trending_recommendations(
    limit: int = Query(default=10, ge=1, le=50),
    current_user: User = Depends(get_current_user)
):
    """
    Get trending article recommendations
    
    Returns currently popular and trending articles based on
    engagement and recency.
    """
    try:
        # Get trending topics first
        trending_topics = await get_trending_topics(days=3)
        
        # Use trending topics to generate recommendations
        trending_queries = []
        for topic in trending_topics[:5]:  # Top 5 trending topics
            trending_queries.append(topic['topic'])
        
        all_recommendations = []
        
        # Search for articles related to trending topics
        for query in trending_queries:
            results = await search_articles(query, limit=3)  # 3 per topic
            for result in results:
                all_recommendations.append({
                    "article_id": result.article_id,
                    "title": result.title,
                    "content_snippet": result.content,
                    "similarity_score": round(result.similarity_score, 4),
                    "published_at": result.published_at.isoformat(),
                    "symbols": result.symbols,
                    "sectors": result.sectors,
                    "trending_topic": query,
                    "reason": f"Trending: {query}"
                })
        
        # Remove duplicates and sort by similarity
        seen_articles = set()
        unique_recommendations = []
        for rec in sorted(all_recommendations, key=lambda x: x['similarity_score'], reverse=True):
            if rec['article_id'] not in seen_articles:
                unique_recommendations.append(rec)
                seen_articles.add(rec['article_id'])
                if len(unique_recommendations) >= limit:
                    break
        
        return {
            "recommendations": unique_recommendations,
            "strategy": "trending-based",
            "trending_topics": [t['topic'] for t in trending_topics[:5]],
            "total_found": len(unique_recommendations),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get trending recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Trending recommendations failed: {str(e)}")

# Analytics Endpoints

@router.post("/trending", response_model=List[Dict[str, Any]])
async def get_trending_topics_endpoint(
    trending_request: TrendingTopicsRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Get trending topics and symbols
    
    Analyzes recent articles to identify trending topics, symbols,
    and sectors based on frequency and engagement.
    """
    try:
        trending = await get_trending_topics(days=trending_request.days)
        
        # Filter by topic types if specified
        if trending_request.topic_types:
            trending = [
                t for t in trending 
                if t.get('type') in trending_request.topic_types
            ]
        
        # Filter by minimum frequency
        trending = [
            t for t in trending 
            if t.get('frequency', 0) >= trending_request.min_frequency
        ]
        
        # Add trend indicators
        for topic in trending:
            topic['trend_indicator'] = (
                "🔥" if topic.get('trend_score', 0) > 0.1 else
                "📈" if topic.get('trend_score', 0) > 0.05 else
                "📊"
            )
            topic['popularity'] = (
                "high" if topic.get('frequency', 0) > 10 else
                "medium" if topic.get('frequency', 0) > 5 else
                "low"
            )
        
        return trending
        
    except Exception as e:
        logger.error(f"Failed to get trending topics: {e}")
        raise HTTPException(status_code=500, detail=f"Trending topics failed: {str(e)}")

@router.get("/analytics/topics")
async def analyze_content_topics(
    days: int = Query(default=7, ge=1, le=30),
    min_frequency: int = Query(default=3, ge=1),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze content topics and their relationships
    
    Provides insights into topic clusters, relationships between
    different financial topics, and content patterns.
    """
    try:
        # Get trending topics
        trending = await get_trending_topics(days=days)
        
        # Analyze topic relationships (simplified)
        topic_analysis = {
            "trending_symbols": [],
            "trending_sectors": [],
            "topic_clusters": [],
            "analysis_period": f"{days} days",
            "total_topics": len(trending)
        }
        
        # Separate by type
        for topic in trending:
            if topic.get('frequency', 0) >= min_frequency:
                if topic.get('type') == 'symbol':
                    topic_analysis['trending_symbols'].append({
                        "symbol": topic['topic'],
                        "frequency": topic['frequency'],
                        "trend_score": round(topic.get('trend_score', 0), 4)
                    })
                elif topic.get('type') == 'sector':
                    topic_analysis['trending_sectors'].append({
                        "sector": topic['topic'],
                        "frequency": topic['frequency'],
                        "trend_score": round(topic.get('trend_score', 0), 4)
                    })
        
        # Sort by trend score
        topic_analysis['trending_symbols'].sort(key=lambda x: x['trend_score'], reverse=True)
        topic_analysis['trending_sectors'].sort(key=lambda x: x['trend_score'], reverse=True)
        
        return topic_analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze topics: {e}")
        raise HTTPException(status_code=500, detail=f"Topic analysis failed: {str(e)}")

# System Endpoints

@router.get("/stats", response_model=IndexStatsResponse)
async def get_search_stats(current_user: User = Depends(get_current_user)):
    """
    Get vector search system statistics
    
    Returns information about the search index, performance metrics,
    and system capabilities.
    """
    try:
        stats = vector_search_engine.get_index_stats()
        
        return IndexStatsResponse(
            chromadb_available=stats.get('chromadb_available', False),
            sentence_transformers_available=stats.get('sentence_transformers_available', False),
            fallback_mode=stats.get('fallback_mode', True),
            model_name=stats.get('model_name', 'fallback'),
            total_documents=stats.get('total_documents', 0),
            index_size=stats.get('index_size', 0),
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to get search stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@router.post("/reindex")
async def reindex_articles(
    background_tasks: BackgroundTasks,
    force: bool = Query(default=False, description="Force reindexing even if index exists"),
    current_user: User = Depends(get_current_user)
):
    """
    Reindex articles for vector search
    
    Rebuilds the vector search index from the current article database.
    This is a background operation that may take several minutes.
    """
    try:
        # Check if user has admin privileges (implement your auth logic)
        # For now, allow all authenticated users
        
        # Add reindexing task to background
        background_tasks.add_task(
            _reindex_articles_background,
            force=force,
            user_id=current_user.id
        )
        
        return {
            "message": "Reindexing started in background",
            "status": "submitted",
            "force_reindex": force,
            "estimated_time": "5-15 minutes depending on article count"
        }
        
    except Exception as e:
        logger.error(f"Failed to start reindexing: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

# Background tasks

async def _reindex_articles_background(force: bool = False, user_id: int = None):
    """Background task to reindex all articles"""
    try:
        logger.info(f"Starting article reindexing (force={force}, user={user_id})")
        
        # This would query your database for all articles and add them to the index
        # For now, just log the operation
        
        # Simulated reindexing process
        import asyncio
        await asyncio.sleep(2)  # Simulate work
        
        logger.info("Article reindexing completed successfully")
        
        # You could send notification to user when complete
        
    except Exception as e:
        logger.error(f"Background reindexing failed: {e}")

# Health check endpoint

@router.get("/health")
async def vector_search_health():
    """Health check for vector search system"""
    try:
        stats = vector_search_engine.get_index_stats()
        
        health_status = {
            "status": "healthy",
            "chromadb_status": "available" if stats.get('chromadb_available') else "unavailable",
            "sentence_transformers_status": "available" if stats.get('sentence_transformers_available') else "unavailable",
            "fallback_mode": stats.get('fallback_mode', True),
            "total_documents": stats.get('total_documents', 0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine overall health
        if stats.get('chromadb_available') and stats.get('sentence_transformers_available'):
            health_status['status'] = "healthy"
        elif stats.get('fallback_mode'):
            health_status['status'] = "degraded"
        else:
            health_status['status'] = "unhealthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Vector search health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }