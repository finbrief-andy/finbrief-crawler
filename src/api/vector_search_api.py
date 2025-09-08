"""
Vector Search API endpoints for semantic search and recommendations.
Provides REST API access to enhanced vector store capabilities.
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from scripts.main import get_db, get_current_user
from src.database.models_migration import User
from src.services.enhanced_vector_store import get_enhanced_vector_store


# Pydantic models for API
class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    content_type: str = Field("news", description="Content type: news, analysis, strategy")
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")
    market_filter: Optional[str] = Field(None, description="Filter by market: global, vn")
    asset_type_filter: Optional[str] = Field(None, description="Filter by asset type")
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity score")


class RecommendationRequest(BaseModel):
    interests: List[str] = Field(..., description="User interests/topics")
    market: Optional[str] = Field(None, description="Market filter")
    limit: int = Field(10, ge=1, le=20, description="Number of recommendations")


class SearchResult(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    similarity: float
    relevance_score: float


class RecommendationResult(BaseModel):
    recommendations: List[SearchResult]
    user_profile: Dict[str, Any]


class RelatedArticlesResponse(BaseModel):
    related_articles: List[SearchResult]
    original_article_id: int


class VectorStoreStats(BaseModel):
    collections: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    cache_stats: Dict[str, Any]
    models: Dict[str, Any]


# Create router
router = APIRouter(prefix="/vector-search", tags=["Vector Search"])


@router.post("/semantic-search", response_model=List[SearchResult])
async def semantic_search(
    request: SemanticSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Perform semantic search across news articles and analysis.
    Uses advanced embedding models to find semantically similar content.
    """
    try:
        vector_store = get_enhanced_vector_store()
        
        if not vector_store.is_available():
            raise HTTPException(
                status_code=503, 
                detail="Vector search not available. ChromaDB/sentence-transformers not installed."
            )
        
        # Build filters
        filters = {}
        if request.market_filter:
            filters["market"] = request.market_filter
        if request.asset_type_filter:
            filters["asset_type"] = request.asset_type_filter
        
        # Perform semantic search
        results = vector_store.semantic_search(
            query=request.query,
            content_type=request.content_type,
            limit=request.limit,
            filters=filters,
            similarity_threshold=request.similarity_threshold
        )
        
        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                id=result["id"],
                text=result["text"],
                metadata=result["metadata"],
                similarity=result["similarity"],
                relevance_score=result["relevance_score"]
            ))
        
        return search_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.get("/related-articles/{news_id}", response_model=RelatedArticlesResponse)
async def get_related_articles(
    news_id: int,
    limit: int = Query(5, ge=1, le=10, description="Number of related articles"),
    market_filter: Optional[str] = Query(None, description="Market filter"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Find articles related to a specific news item using semantic similarity.
    """
    try:
        vector_store = get_enhanced_vector_store()
        
        if not vector_store.is_available():
            raise HTTPException(
                status_code=503,
                detail="Vector search not available."
            )
        
        # Find related articles
        related = vector_store.find_related_articles(
            news_id=news_id,
            limit=limit,
            market_filter=market_filter
        )
        
        # Convert to response format
        related_results = []
        for item in related:
            related_results.append(SearchResult(
                id=item["id"],
                text=item["text"],
                metadata=item["metadata"],
                similarity=item["similarity"],
                relevance_score=item["relevance_score"]
            ))
        
        return RelatedArticlesResponse(
            related_articles=related_results,
            original_article_id=news_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Related articles search failed: {str(e)}")


@router.post("/recommendations", response_model=RecommendationResult)
async def get_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate personalized recommendations based on user interests.
    Uses semantic similarity to find relevant content.
    """
    try:
        vector_store = get_enhanced_vector_store()
        
        if not vector_store.is_available():
            raise HTTPException(
                status_code=503,
                detail="Recommendation service not available."
            )
        
        # Generate recommendations
        recs = vector_store.generate_recommendations(
            user_interests=request.interests,
            market=request.market,
            limit=request.limit
        )
        
        # Convert to response format
        recommendations = []
        for rec in recs:
            recommendations.append(SearchResult(
                id=rec["id"],
                text=rec["text"],
                metadata=rec["metadata"],
                similarity=rec["similarity"],
                relevance_score=rec["relevance_score"]
            ))
        
        # Create user profile summary
        user_profile = {
            "interests": request.interests,
            "market_focus": request.market,
            "recommendation_count": len(recommendations),
            "generated_at": "now"
        }
        
        return RecommendationResult(
            recommendations=recommendations,
            user_profile=user_profile
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


@router.get("/cluster-analysis")
async def get_cluster_analysis(
    content_type: str = Query("news", description="Content type to cluster"),
    n_clusters: int = Query(10, ge=2, le=20, description="Number of clusters"),
    current_user: User = Depends(get_current_user)
):
    """
    Perform clustering analysis to discover content topics and themes.
    """
    try:
        vector_store = get_enhanced_vector_store()
        
        if not vector_store.is_available():
            raise HTTPException(
                status_code=503,
                detail="Clustering analysis not available."
            )
        
        # Perform clustering
        cluster_results = vector_store.cluster_similar_content(
            content_type=content_type,
            n_clusters=n_clusters
        )
        
        if not cluster_results:
            raise HTTPException(
                status_code=404,
                detail="No content available for clustering"
            )
        
        return cluster_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster analysis failed: {str(e)}")


@router.get("/stats", response_model=VectorStoreStats)
async def get_vector_store_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive statistics about the vector store.
    """
    try:
        vector_store = get_enhanced_vector_store()
        
        if not vector_store.is_available():
            raise HTTPException(
                status_code=503,
                detail="Vector store not available."
            )
        
        stats = vector_store.get_collection_stats()
        
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        
        return VectorStoreStats(
            collections=stats["collections"],
            performance_metrics=stats["performance_metrics"],
            cache_stats=stats["cache_stats"],
            models=stats["models"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@router.post("/backfill-embeddings")
async def backfill_embeddings(
    limit: int = Query(1000, ge=1, le=5000, description="Number of items to backfill"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Backfill embeddings for existing news and analysis data.
    Admin-only endpoint for maintenance.
    """
    # Check if user is admin
    if current_user.role.value not in ["admin", "system"]:
        raise HTTPException(
            status_code=403,
            detail="Only admins can trigger embedding backfill"
        )
    
    try:
        vector_store = get_enhanced_vector_store()
        
        if not vector_store.is_available():
            raise HTTPException(
                status_code=503,
                detail="Vector store not available for backfill."
            )
        
        # Trigger backfill
        vector_store.backfill_embeddings(db, limit=limit)
        
        return {
            "status": "success",
            "message": f"Backfill initiated for up to {limit} items",
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backfill failed: {str(e)}")


@router.delete("/cleanup-embeddings")
async def cleanup_old_embeddings(
    days_old: int = Query(90, ge=7, le=365, description="Age threshold in days"),
    current_user: User = Depends(get_current_user)
):
    """
    Clean up old embeddings to manage storage.
    Admin-only endpoint for maintenance.
    """
    # Check if user is admin
    if current_user.role.value not in ["admin", "system"]:
        raise HTTPException(
            status_code=403,
            detail="Only admins can cleanup embeddings"
        )
    
    try:
        vector_store = get_enhanced_vector_store()
        
        if not vector_store.is_available():
            raise HTTPException(
                status_code=503,
                detail="Vector store not available for cleanup."
            )
        
        # Trigger cleanup
        vector_store.cleanup_old_embeddings(days_old=days_old)
        
        return {
            "status": "success",
            "message": f"Cleanup completed for embeddings older than {days_old} days",
            "days_old": days_old
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# Health check endpoint
@router.get("/health")
async def vector_search_health():
    """
    Check vector search service health.
    """
    try:
        vector_store = get_enhanced_vector_store()
        
        health_status = {
            "status": "healthy" if vector_store.is_available() else "unavailable",
            "vector_store_available": vector_store.is_available(),
            "chroma_available": vector_store.client is not None,
            "embedding_models_loaded": vector_store.embedding_model is not None,
            "timestamp": "now"
        }
        
        if vector_store.is_available():
            stats = vector_store.get_collection_stats()
            health_status["collections"] = {
                "news_count": stats.get("collections", {}).get("news", {}).get("count", 0),
                "analysis_count": stats.get("collections", {}).get("analysis", {}).get("count", 0)
            }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "now"
        }