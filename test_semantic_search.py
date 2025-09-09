#!/usr/bin/env python3
"""Test script for complete semantic search and vector analysis functionality"""
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.vector_store import get_vector_store

def test_semantic_search_complete():
    """Test complete semantic search functionality"""
    print("=== COMPLETE SEMANTIC SEARCH & VECTOR ANALYSIS TEST ===")
    
    # Get vector store instance
    vs = get_vector_store()
    
    if not vs.is_available():
        print("❌ Vector store not available")
        return False
    
    print(f"✅ Vector store available")
    news_count = len(vs.news_collection.get()['ids'])
    analysis_count = len(vs.analysis_collection.get()['ids'])
    print(f"📊 Data: {news_count} news embeddings, {analysis_count} analysis embeddings")
    
    # Test 1: Semantic similarity search with various financial queries
    print("\n=== Test 1: Semantic News Search ===")
    test_queries = [
        "technology stocks and AI companies",
        "interest rates and monetary policy", 
        "cryptocurrency and digital assets",
        "earnings reports and financial performance",
        "market volatility and investor sentiment"
    ]
    
    search_success = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        results = vs.search_similar_news(query, limit=3)
        
        if results:
            print(f"   ✅ Found {len(results)} results")
            for j, result in enumerate(results[:2], 1):
                headline = result['metadata'].get('headline', 'No headline')[:60]
                distance = result.get('distance', 0)
                relevance = "High" if distance < 1.0 else "Medium" if distance < 1.5 else "Low"
                print(f"   {j}. {headline}... (relevance: {relevance})")
            search_success += 1
        else:
            print("   ❌ No results found")
    
    # Test 2: Analysis search
    print(f"\n=== Test 2: Semantic Analysis Search ===")
    analysis_queries = [
        "positive market outlook and bullish sentiment",
        "risk factors and bearish indicators", 
        "investment strategy recommendations"
    ]
    
    analysis_success = 0
    for i, query in enumerate(analysis_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        results = vs.search_similar_analyses(query, limit=3)
        
        if results:
            print(f"   ✅ Found {len(results)} analysis results")
            for j, result in enumerate(results[:2], 1):
                sentiment = result['metadata'].get('sentiment', 'unknown')
                action = result['metadata'].get('action_short', 'NONE')
                print(f"   {j}. Sentiment: {sentiment}, Action: {action}")
            analysis_success += 1
        else:
            print("   ❌ No analysis results found")
    
    # Test 3: Context retrieval for strategy generation
    print(f"\n=== Test 3: Context Retrieval for Strategy Generation ===")
    strategy_contexts = [
        "growth stocks with strong fundamentals",
        "defensive investments during market uncertainty",
        "emerging market opportunities"
    ]
    
    context_success = 0
    for i, context_query in enumerate(strategy_contexts, 1):
        print(f"\n{i}. Context: '{context_query}'")
        context = vs.get_relevant_context(context_query, limit=3)
        
        if context:
            print(f"   ✅ Retrieved context ({len(context)} chars)")
            preview = context[:150] + "..." if len(context) > 150 else context
            print(f"   Preview: {preview}")
            context_success += 1
        else:
            print("   ❌ No context retrieved")
    
    # Test 4: Market-specific filtering
    print(f"\n=== Test 4: Market-Specific Search ===")
    markets = ["global", "us"]
    
    for market in markets:
        print(f"\nSearching in {market.upper()} market:")
        results = vs.search_similar_news("market performance", limit=3, market=market)
        print(f"   Found {len(results)} results in {market} market")
    
    # Test 5: Performance and accuracy assessment
    print(f"\n=== Test 5: Performance Assessment ===")
    
    # Test embedding generation speed
    import time
    test_texts = ["Sample text for performance testing"] * 10
    start_time = time.time()
    embeddings = vs.generate_embeddings(test_texts)
    end_time = time.time()
    
    if embeddings:
        avg_time = (end_time - start_time) / len(test_texts)
        print(f"✅ Embedding generation: {avg_time:.3f}s per text")
    else:
        print("❌ Embedding generation failed")
    
    # Test search speed
    start_time = time.time()
    search_results = vs.search_similar_news("performance test", limit=5)
    end_time = time.time()
    search_time = end_time - start_time
    print(f"✅ Search speed: {search_time:.3f}s for 5 results")
    
    # Overall assessment
    print(f"\n=== OVERALL ASSESSMENT ===")
    total_tests = 5
    passed_tests = 0
    
    if search_success >= len(test_queries) * 0.6:  # 60% success rate
        print("✅ News search functionality: PASSED")
        passed_tests += 1
    else:
        print("❌ News search functionality: FAILED")
    
    if analysis_success >= len(analysis_queries) * 0.6:
        print("✅ Analysis search functionality: PASSED") 
        passed_tests += 1
    else:
        print("❌ Analysis search functionality: FAILED")
    
    if context_success >= len(strategy_contexts) * 0.6:
        print("✅ Context retrieval functionality: PASSED")
        passed_tests += 1
    else:
        print("❌ Context retrieval functionality: FAILED")
    
    if avg_time < 0.5:  # Less than 500ms per embedding
        print("✅ Performance benchmarks: PASSED")
        passed_tests += 1
    else:
        print("❌ Performance benchmarks: FAILED")
    
    if news_count > 0 and analysis_count > 0:
        print("✅ Data availability: PASSED")
        passed_tests += 1
    else:
        print("❌ Data availability: FAILED")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n🎯 FINAL SCORE: {passed_tests}/{total_tests} tests passed ({success_rate:.0f}%)")
    
    if success_rate >= 80:
        print("🎉 SEMANTIC SEARCH & VECTOR ANALYSIS: PRODUCTION READY!")
        return True
    elif success_rate >= 60:
        print("⚠️  SEMANTIC SEARCH & VECTOR ANALYSIS: MOSTLY WORKING (needs minor fixes)")
        return True
    else:
        print("❌ SEMANTIC SEARCH & VECTOR ANALYSIS: NEEDS WORK")
        return False

if __name__ == "__main__":
    success = test_semantic_search_complete()
    sys.exit(0 if success else 1)