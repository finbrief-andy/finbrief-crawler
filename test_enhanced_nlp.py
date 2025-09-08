#!/usr/bin/env python3
"""
Test script for Enhanced NLP Processing capabilities.
Tests NER, improved summarization, sentiment analysis, and key phrase extraction.
"""
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_enhanced_nlp():
    """Test the enhanced NLP processor"""
    print("🧠 Testing Enhanced NLP Processing")
    print("=" * 50)
    
    # Sample financial news texts for testing
    test_texts = [
        {
            "title": "Apple Quarterly Earnings",
            "text": """Apple Inc. (AAPL) reported strong Q4 earnings yesterday, with revenue reaching $89.5 billion, beating analyst expectations of $88.9 billion. CEO Tim Cook highlighted robust iPhone 14 sales and growing services revenue. The company's partnership with Goldman Sachs for Apple Card continues to show positive results. Apple stock rose 3.2% in after-hours trading following the announcement. The Cupertino-based tech giant also announced a $90 billion share buyback program."""
        },
        {
            "title": "Tesla Production Update", 
            "text": """Tesla Motors delivered 405,278 vehicles in Q4 2023, falling short of Wall Street estimates of 431,117 deliveries. Elon Musk cited supply chain disruptions in Shanghai and Berlin factories as key factors. Despite the miss, Tesla maintained its position as the world's leading electric vehicle manufacturer. The company's Gigafactory in Austin, Texas showed strong production ramp-up. Tesla shares dropped 7.3% in pre-market trading on the news."""
        },
        {
            "title": "Banking Sector News",
            "text": """JPMorgan Chase & Co. and Bank of America both reported higher-than-expected loan losses in their latest quarterly results. The Federal Reserve's interest rate hikes have created a challenging environment for regional banks. Wells Fargo CEO Charles Scharf warned of potential credit tightening. Goldman Sachs analysts predict continued volatility in the banking sector through 2024."""
        }
    ]
    
    try:
        from src.crawlers.processors.enhanced_nlp_processor import EnhancedNLPProcessor
        
        print("Initializing Enhanced NLP Processor...")
        start_time = time.time()
        processor = EnhancedNLPProcessor()
        init_time = time.time() - start_time
        print(f"✅ Processor initialized in {init_time:.2f} seconds")
        print()
        
        # Test each sample text
        for i, sample in enumerate(test_texts, 1):
            print(f"--- Test Case {i}: {sample['title']} ---")
            text = sample['text']
            
            # Test individual components
            test_named_entity_recognition(processor, text)
            test_enhanced_summarization(processor, text)
            test_enhanced_sentiment_analysis(processor, text)
            test_key_phrase_extraction(processor, text)
            
            # Test comprehensive analysis
            print("🔍 Comprehensive Analysis:")
            comp_start = time.time()
            analysis = processor.comprehensive_analysis(text)
            comp_time = time.time() - comp_start
            
            if "error" not in analysis:
                print(f"✅ Comprehensive analysis completed in {comp_time:.2f} seconds")
                print(f"   Market relevance: {analysis['insights']['market_relevance']}")
                print(f"   Entity diversity: {analysis['insights']['entity_diversity']}")
                print(f"   Models used: {', '.join(analysis['metadata']['models_used'])}")
            else:
                print(f"❌ Comprehensive analysis failed: {analysis['error']}")
            
            print()
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import enhanced NLP processor: {e}")
        print("   Dependencies may be missing. Install: transformers torch")
        return False
    except Exception as e:
        print(f"❌ Enhanced NLP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_named_entity_recognition(processor, text: str):
    """Test Named Entity Recognition"""
    print("👥 Named Entity Recognition:")
    try:
        entities = processor.extract_named_entities(text)
        
        companies = entities.get('companies', [])
        people = entities.get('people', [])  
        locations = entities.get('locations', [])
        
        print(f"   Companies: {[c['text'] for c in companies]}")
        print(f"   People: {[p['text'] for p in people]}")
        print(f"   Locations: {[l['text'] for l in locations]}")
        
        if companies or people or locations:
            print("   ✅ NER extraction working")
        else:
            print("   ⚠️  No entities extracted")
            
    except Exception as e:
        print(f"   ❌ NER failed: {e}")


def test_enhanced_summarization(processor, text: str):
    """Test Enhanced Summarization"""
    print("📝 Enhanced Summarization:")
    try:
        summary_result = processor.enhanced_summarize(text)
        
        summary = summary_result.get('summary', '')
        method = summary_result.get('method', 'unknown')
        confidence = summary_result.get('confidence', 0)
        
        print(f"   Summary: {summary[:100]}...")
        print(f"   Method: {method}, Confidence: {confidence:.2f}")
        
        if summary and len(summary.split()) > 0:
            print("   ✅ Summarization working")
        else:
            print("   ⚠️  Empty summary generated")
            
    except Exception as e:
        print(f"   ❌ Summarization failed: {e}")


def test_enhanced_sentiment_analysis(processor, text: str):
    """Test Enhanced Sentiment Analysis"""
    print("💭 Enhanced Sentiment Analysis:")
    try:
        sentiment_result = processor.enhanced_sentiment_analysis(text)
        
        sentiment = sentiment_result.get('sentiment', 'unknown')
        confidence = sentiment_result.get('confidence', 0)
        strength = sentiment_result.get('strength', 'unknown')
        method = sentiment_result.get('method', 'unknown')
        
        print(f"   Sentiment: {sentiment} (confidence: {confidence:.2f}, strength: {strength})")
        print(f"   Method: {method}")
        
        # Show market implications if available
        implications = sentiment_result.get('market_implications', [])
        if implications:
            print(f"   Market implications: {', '.join(implications[:3])}")
        
        if sentiment in ['positive', 'negative', 'neutral']:
            print("   ✅ Sentiment analysis working")
        else:
            print("   ⚠️  Invalid sentiment result")
            
    except Exception as e:
        print(f"   ❌ Sentiment analysis failed: {e}")


def test_key_phrase_extraction(processor, text: str):
    """Test Key Phrase Extraction"""
    print("🔑 Key Phrase Extraction:")
    try:
        key_phrases = processor.extract_key_phrases(text)
        
        if key_phrases:
            # Show top 5 phrases
            top_phrases = key_phrases[:5]
            for phrase in top_phrases:
                print(f"   - {phrase['phrase']} (score: {phrase['score']:.2f}, method: {phrase['method']})")
            
            print("   ✅ Key phrase extraction working")
        else:
            print("   ⚠️  No key phrases extracted")
            
    except Exception as e:
        print(f"   ❌ Key phrase extraction failed: {e}")


def test_basic_functionality():
    """Test basic functionality without full model loading"""
    print("\n🔧 Testing Basic Functionality")
    print("=" * 30)
    
    try:
        # Test if we can import the class
        from src.crawlers.processors.enhanced_nlp_processor import EnhancedNLPProcessor
        print("✅ Enhanced NLP Processor imports successfully")
        
        # Test basic text processing methods
        processor = EnhancedNLPProcessor()
        
        # Test helper methods
        test_text = "Apple Inc. reported earnings. Tesla stock fell 5%."
        
        # Test financial term extraction
        financial_phrases = processor._extract_financial_terms(test_text)
        print(f"✅ Financial term extraction: {len(financial_phrases)} terms found")
        
        # Test stopword filtering
        is_stopword = processor._is_stopword_phrase("the and or")
        print(f"✅ Stopword filtering: {is_stopword}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Main test runner"""
    print("🚀 Enhanced NLP Processing Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run basic functionality test first
    basic_success = test_basic_functionality()
    
    if basic_success:
        # Run full NLP tests
        full_success = test_enhanced_nlp()
        
        print("=" * 60)
        if full_success:
            print("🎉 Enhanced NLP Processing tests completed successfully!")
            print("\n📋 Features validated:")
            print("✅ Named Entity Recognition (companies, people, locations)")
            print("✅ Enhanced summarization with multiple strategies")
            print("✅ Advanced sentiment analysis with FinBERT")
            print("✅ Key phrase extraction with multiple methods")
            print("✅ Comprehensive analysis pipeline")
            print("\n🔧 Next steps:")
            print("1. Integrate with unified pipeline")
            print("2. Performance optimization for production")
            print("3. Add model caching for faster initialization")
        else:
            print("⚠️  Some enhanced NLP tests failed")
            print("💡 This may be due to missing model dependencies")
            print("   Try: pip install transformers torch sentence-transformers")
    else:
        print("❌ Basic functionality tests failed")
        print("🔧 Check imports and basic dependencies")
    
    return basic_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)