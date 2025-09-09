"""
Semantic Search API endpoints.
Provides semantic similarity search and related article recommendations.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from src.services.vector_store import get_vector_store
from src.api.auth import get_current_user
from src.database.models_migration import UserBase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/semantic", tags=["semantic_search"])


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, ge=1, le=100, description="Number of results to return")
    market: Optional[str] = Field(None, description="Filter by market (us, vn, global)")
    asset_type: Optional[str] = Field(None, description="Filter by asset type (stocks, crypto, etc)")


class SemanticSearchResult(BaseModel):
    """Result model for semantic search"""
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search"""
    query: str
    results: List[SemanticSearchResult]
    total_found: int
    search_type: str


class RecommendationsRequest(BaseModel):
    """Request model for article recommendations"""
    article_id: str = Field(..., description="ID of the article to find similar articles for")
    limit: int = Field(5, ge=1, le=20, description="Number of recommendations to return")


@router.post("/search/news", response_model=SemanticSearchResponse)
async def search_similar_news(
    request: SemanticSearchRequest,
    current_user: UserBase = Depends(get_current_user)
):
    """
    Search for semantically similar news articles
    """
    try:
        vs = get_vector_store()
        
        if not vs.is_available():
            raise HTTPException(
                status_code=503, 
                detail="Vector search service is not available. Please contact administrator."
            )
        
        # Perform semantic search
        results = vs.search_similar_news(
            query=request.query,
            limit=request.limit,
            market=request.market,
            asset_type=request.asset_type
        )
        
        # Format results
        search_results = [
            SemanticSearchResult(
                id=result["id"],
                text=result["text"],
                metadata=result["metadata"],
                distance=result.get("distance")
            )
            for result in results
        ]
        
        logger.info(f"User {current_user.username} searched news: '{request.query}' -> {len(search_results)} results")
        
        return SemanticSearchResponse(
            query=request.query,
            results=search_results,
            total_found=len(search_results),
            search_type="news"
        )
        
    except Exception as e:
        logger.error(f"Error in semantic news search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/search/analysis", response_model=SemanticSearchResponse)
async def search_similar_analysis(
    request: SemanticSearchRequest,
    current_user: UserBase = Depends(get_current_user)
):
    """
    Search for semantically similar analysis results
    """
    try:
        vs = get_vector_store()
        
        if not vs.is_available():
            raise HTTPException(
                status_code=503, 
                detail="Vector search service is not available. Please contact administrator."
            )
        
        # Perform semantic search
        results = vs.search_similar_analyses(
            query=request.query,
            limit=request.limit
        )
        
        # Format results
        search_results = [
            SemanticSearchResult(
                id=result["id"],
                text=result["text"],
                metadata=result["metadata"],
                distance=result.get("distance")
            )
            for result in results
        ]
        
        logger.info(f"User {current_user.username} searched analysis: '{request.query}' -> {len(search_results)} results")
        
        return SemanticSearchResponse(
            query=request.query,
            results=search_results,
            total_found=len(search_results),
            search_type="analysis"
        )
        
    except Exception as e:
        logger.error(f"Error in semantic analysis search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search/news")
async def search_news_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Number of results"),
    market: Optional[str] = Query(None, description="Filter by market"),
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
    current_user: UserBase = Depends(get_current_user)
):
    """
    GET endpoint for semantic news search (for easy URL-based queries)
    """
    request = SemanticSearchRequest(
        query=q,
        limit=limit,
        market=market,
        asset_type=asset_type
    )
    return await search_similar_news(request, current_user)


@router.post("/recommendations", response_model=SemanticSearchResponse)
async def get_article_recommendations(
    request: RecommendationsRequest,
    current_user: UserBase = Depends(get_current_user)
):
    """
    Get recommendations for similar articles based on an existing article
    """
    try:
        vs = get_vector_store()
        
        if not vs.is_available():
            raise HTTPException(
                status_code=503, 
                detail="Vector search service is not available. Please contact administrator."
            )
        
        # First, try to get the article content to use as query
        # For now, we'll use the article ID directly as a limitation
        # In a full implementation, we'd fetch the article from the database
        
        # Simple implementation: search for similar articles using ID
        # This could be enhanced to fetch article content and use that as query
        query_text = f"article {request.article_id}"
        
        results = vs.search_similar_news(
            query=query_text,
            limit=request.limit + 1  # +1 to potentially exclude the original
        )
        
        # Filter out the original article if present
        filtered_results = [
            r for r in results 
            if r["id"] != f"news_{request.article_id}"
        ][:request.limit]
        
        # Format results
        search_results = [
            SemanticSearchResult(
                id=result["id"],
                text=result["text"],
                metadata=result["metadata"],
                distance=result.get("distance")
            )
            for result in filtered_results
        ]
        
        logger.info(f"User {current_user.username} got recommendations for article {request.article_id}: {len(search_results)} results")
        
        return SemanticSearchResponse(
            query=f"Recommendations for article {request.article_id}",
            results=search_results,
            total_found=len(search_results),
            search_type="recommendations"
        )
        
    except Exception as e:
        logger.error(f"Error getting article recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


@router.get("/status")
async def get_semantic_search_status(
    current_user: UserBase = Depends(get_current_user)
):
    """
    Get status of semantic search service
    """
    vs = get_vector_store()
    
    if not vs.is_available():
        return {
            "status": "unavailable",
            "message": "Vector search service is not available",
            "news_embeddings": 0,
            "analysis_embeddings": 0
        }
    
    try:
        news_count = len(vs.news_collection.get()['ids'])
        analysis_count = len(vs.analysis_collection.get()['ids'])
        
        return {
            "status": "available",
            "message": "Vector search service is running",
            "news_embeddings": news_count,
            "analysis_embeddings": analysis_count,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384
        }
        
    except Exception as e:
        logger.error(f"Error getting semantic search status: {e}")
        return {
            "status": "error",
            "message": f"Error checking status: {str(e)}",
            "news_embeddings": 0,
            "analysis_embeddings": 0
        }