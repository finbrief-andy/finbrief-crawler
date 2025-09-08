# Vector Search & Semantic Analysis Guide

## Overview
FinBrief's enhanced vector search system provides advanced semantic search capabilities, content recommendations, and similarity analysis using ChromaDB and sentence transformers. The system enables intelligent content discovery beyond traditional keyword matching.

## Architecture

### Core Components

1. **Enhanced Vector Store** (`enhanced_vector_store.py`)
   - ChromaDB integration with optimized settings
   - Multiple embedding models (general + financial-specific)
   - Advanced semantic search with filtering
   - Content clustering and topic discovery
   - Performance optimization with caching

2. **Vector Search API** (`vector_search_api.py`)  
   - REST API endpoints for semantic operations
   - User-friendly search interface
   - Recommendation engine
   - Administrative tools

3. **Backward Compatibility Layer**
   - Maintains original `vector_store.py` interface
   - Seamless upgrade path
   - Graceful degradation when dependencies unavailable

## Features

### 1. Semantic Search
Find content by meaning rather than exact keyword matches:

```python
from src.services.enhanced_vector_store import get_enhanced_vector_store

store = get_enhanced_vector_store()
results = store.semantic_search(
    query="Apple quarterly earnings impact",
    content_type="news",
    limit=10,
    filters={"market": "global"},
    similarity_threshold=0.3
)
```

**Capabilities:**
- Multi-model embeddings (general + financial)
- Advanced filtering by market, asset type, date
- Similarity threshold controls
- Relevance scoring with metadata signals
- Caching for performance

### 2. Content Recommendations
Personalized content discovery based on user interests:

```python
recommendations = store.generate_recommendations(
    user_interests=["technology stocks", "AI companies", "earnings reports"],
    market="global",
    limit=10
)
```

**Features:**
- Interest-based content matching
- Source and topic diversification
- Market-specific recommendations
- User profile analysis

### 3. Related Article Discovery
Find articles similar to a specific news item:

```python
related = store.find_related_articles(
    news_id=123,
    limit=5,
    market_filter="global"
)
```

### 4. Content Clustering
Automatic topic discovery and content organization:

```python
clusters = store.cluster_similar_content(
    content_type="news",
    n_clusters=10
)
```

**Analysis Capabilities:**
- K-means clustering of content
- Topic extraction from clusters
- Content theme identification
- Trend analysis support

## API Endpoints

### Semantic Search
```
POST /vector-search/semantic-search
```
Perform semantic search across content collections.

**Request:**
```json
{
    "query": "Federal Reserve interest rate decision",
    "content_type": "news",
    "limit": 10,
    "market_filter": "global",
    "similarity_threshold": 0.4
}
```

**Response:**
```json
[
    {
        "id": "news_123",
        "text": "Federal Reserve announces rate hike...",
        "metadata": {"source": "reuters", "published_at": "2024-01-15"},
        "similarity": 0.87,
        "relevance_score": 0.92
    }
]
```

### Related Articles
```
GET /vector-search/related-articles/{news_id}?limit=5
```
Find articles related to a specific news item.

### Recommendations  
```
POST /vector-search/recommendations
```
Generate personalized content recommendations.

**Request:**
```json
{
    "interests": ["technology", "artificial intelligence", "earnings"],
    "market": "global",
    "limit": 10
}
```

### Cluster Analysis
```
GET /vector-search/cluster-analysis?content_type=news&n_clusters=10
```
Discover content topics and themes through clustering.

### Statistics & Health
```
GET /vector-search/stats
GET /vector-search/health
```
Monitor system performance and collection statistics.

## Embedding Models

### General Purpose Model
- **Model**: `all-MiniLM-L6-v2` (default)
- **Size**: ~90MB
- **Languages**: Multilingual support
- **Performance**: Fast inference, good quality

### Financial Specialized Model
- **Model**: `paraphrase-MiniLM-L6-v2`
- **Purpose**: Financial content optimization
- **Usage**: Automatic for financial terms and contexts

### Model Selection
The system automatically chooses the appropriate model:
- Financial model for market-related content
- General model for broader content
- Fallback mechanisms for model availability

## Data Collections

### News Embeddings
- **Collection**: `news_embeddings_v2`
- **Content**: Article headlines + summaries
- **Metadata**: Source, market, tickers, urgency scores
- **Indexing**: Semantic similarity + metadata filters

### Analysis Embeddings
- **Collection**: `analysis_embeddings_v2`
- **Content**: Analysis rationale + sentiment context
- **Metadata**: Actions, confidence, sentiment scores
- **Usage**: Strategy-related searches

### Strategy Embeddings
- **Collection**: `strategy_embeddings`
- **Content**: Investment strategies and insights
- **Purpose**: Strategy recommendation and comparison

### Entity Mentions
- **Collection**: `entity_mentions`
- **Content**: Company and entity references
- **Purpose**: Entity-based content discovery

## Performance Optimization

### Caching Strategy
- **Embedding Cache**: 1000 most recent embeddings
- **Cache Hit Rate**: Tracked in performance metrics
- **Memory Management**: LRU eviction policy

### Batch Processing
- **Embedding Generation**: Vectorized operations
- **Search Operations**: Optimized ChromaDB queries
- **Model Warmup**: Preload models for faster response

### Storage Optimization
- **ChromaDB Settings**: Optimized for production
- **Data Retention**: Automatic cleanup of old embeddings
- **Compression**: Efficient vector storage

## Configuration

### Environment Variables
```bash
# Vector storage
VECTOR_STORE_PATH=./data/vectors

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Performance
EMBEDDING_CACHE_SIZE=1000
MAX_SEARCH_RESULTS=100

# Cleanup
EMBEDDING_RETENTION_DAYS=90
```

### Production Settings
```python
# ChromaDB optimization
settings = Settings(
    persist_directory="./data/vectors",
    anonymized_telemetry=False,
    allow_reset=False
)

# Model configuration  
store = EnhancedVectorStore(
    persist_directory="./data/vectors",
    embedding_model="all-MiniLM-L6-v2"
)
```

## Installation & Setup

### Dependencies
```bash
# Required packages
pip install chromadb sentence-transformers scikit-learn

# Optional GPU support
pip install torch[cuda]  # For NVIDIA GPUs
```

### Initial Setup
```bash
# Create storage directory
mkdir -p ./data/vectors

# Initialize collections (automatic on first use)
python -c "from src.services.enhanced_vector_store import get_enhanced_vector_store; get_enhanced_vector_store()"

# Backfill existing data
curl -X POST "http://localhost:8000/vector-search/backfill-embeddings?limit=1000" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Verification
```bash
# Check system health
curl "http://localhost:8000/vector-search/health"

# View statistics
curl "http://localhost:8000/vector-search/stats" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Integration Examples

### Add to News Pipeline
```python
# In your news processing pipeline
from src.services.enhanced_vector_store import get_enhanced_vector_store

store = get_enhanced_vector_store()
if store.is_available():
    store.add_news_embedding(news_item)
```

### Search Integration
```python
# In your search service
results = store.semantic_search(
    query=user_query,
    content_type="news",
    filters={"market": user_market},
    limit=10
)
```

### Recommendation Service
```python
# User preference-based recommendations
user_interests = ["technology", "earnings", "AI"]
recommendations = store.generate_recommendations(
    user_interests=user_interests,
    market=user_preferred_market,
    limit=20
)
```

## Monitoring & Maintenance

### Performance Metrics
- **Embeddings Generated**: Total embedding operations
- **Searches Performed**: Number of search queries  
- **Cache Hit Rate**: Embedding cache efficiency
- **Recommendations Served**: Recommendation system usage

### Health Monitoring
```python
# Check system health
status = store.get_collection_stats()
print(f"News items: {status['collections']['news']['count']}")
print(f"Cache hit rate: {status['cache_stats']['hit_rate']:.2%}")
```

### Maintenance Tasks
```python
# Clean up old embeddings (admin only)
store.cleanup_old_embeddings(days_old=90)

# Backfill new content
store.backfill_embeddings(session, limit=1000)

# Performance optimization
store.model_metrics  # View performance statistics
```

## Troubleshooting

### Common Issues

1. **ChromaDB Not Available**
   - Install: `pip install chromadb`
   - Check permissions for data directory
   - Verify disk space availability

2. **Sentence Transformers Missing**
   - Install: `pip install sentence-transformers`
   - First run downloads models (~200MB)
   - Check internet connectivity

3. **Slow Performance**
   - Enable GPU support if available
   - Increase cache size for frequent searches
   - Use batch operations for bulk processing

4. **Out of Memory Errors**
   - Reduce batch sizes
   - Use CPU-only models for limited RAM
   - Implement model quantization

### Debug Information
```python
# System diagnostics
store = get_enhanced_vector_store()
print(f"Available: {store.is_available()}")
print(f"Models loaded: {store._get_active_models()}")
print(f"Collections: {list(store.client.list_collections())}")
```

## Migration Guide

### From Original Vector Store
The enhanced system maintains backward compatibility:

```python
# Old code continues to work
from src.services.vector_store import get_vector_store
store = get_vector_store()  # Returns enhanced store

# New features available
store.semantic_search(...)  # Enhanced search
store.generate_recommendations(...)  # New capability
```

### Data Migration
Existing embeddings are automatically compatible. New features require re-embedding:

```bash
# Migrate existing data
curl -X POST "/vector-search/backfill-embeddings?limit=5000"
```

## Performance Benchmarks

### Typical Performance
- **Embedding Generation**: ~100 texts/second (CPU)
- **Search Response**: <100ms for 10 results
- **Memory Usage**: ~500MB with models loaded
- **Storage**: ~1KB per embedded text

### Scaling Recommendations
- **Small deployment**: <10K documents, single instance
- **Medium deployment**: 10K-100K documents, consider GPU
- **Large deployment**: >100K documents, distributed setup

---

*Last Updated: January 2025*
*Status: Production Ready - Enhanced Vector Search System*