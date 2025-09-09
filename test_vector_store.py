#!/usr/bin/env python3
"""Test script for vector store functionality"""
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.vector_store import VectorStore, get_vector_store

def test_vector_store():
    """Test vector store functionality"""
    print("Testing vector store functionality...")
    
    # Test direct initialization
    print("\n=== Testing VectorStore Initialization ===")
    vs = VectorStore()
    
    if vs.is_available():
        print("✅ Vector store is available")
        print(f"News collection has {len(vs.news_collection.get()['ids'])} embeddings")
        print(f"Analysis collection has {len(vs.analysis_collection.get()['ids'])} embeddings")
    else:
        print("❌ Vector store not available")
        return
    
    # Test embedding generation
    print("\n=== Testing Embedding Generation ===")
    test_texts = [
        "Apple stock rises after earnings beat expectations",
        "Gold prices surge due to inflation concerns", 
        "Tesla announces new manufacturing facility",
        "Federal Reserve raises interest rates"
    ]
    
    embeddings = vs.generate_embeddings(test_texts)
    if embeddings:
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
    else:
        print("❌ Failed to generate embeddings")
        return
    
    # Test semantic search
    print("\n=== Testing Semantic Search ===")
    test_queries = [
        "technology stocks performance",
        "precious metals investment",
        "central bank policy changes"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = vs.search_similar_news(query, limit=3)
        if results:
            print(f"  Found {len(results)} similar items")
            for i, result in enumerate(results[:2]):
                headline = result['metadata'].get('headline', 'No headline')[:60]
                distance = result.get('distance', 'N/A')
                print(f"    {i+1}. {headline}... (distance: {distance})")
        else:
            print("  No results found")
    
    # Test context retrieval
    print("\n=== Testing Context Retrieval ===")
    context = vs.get_relevant_context("stock market volatility", limit=3)
    if context:
        print("✅ Retrieved relevant context:")
        print(context[:200] + "..." if len(context) > 200 else context)
    else:
        print("No context retrieved")
    
    # Test singleton pattern
    print("\n=== Testing Singleton Pattern ===")
    vs2 = get_vector_store()
    if vs2 is vs:
        print("✅ Singleton pattern working correctly")
    else:
        print("⚠️  Singleton pattern not working as expected")

if __name__ == "__main__":
    test_vector_store()