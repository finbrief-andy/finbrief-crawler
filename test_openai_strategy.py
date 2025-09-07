#!/usr/bin/env python3
"""
Test script for OpenAI-powered strategy generation
"""
import os
import sys
sys.path.append('.')

from src.services.strategy_generator import StrategyGenerator
from src.database.models_migration import init_db_and_create, StrategyHorizonEnum, MarketEnum
from sqlalchemy.orm import sessionmaker

def test_strategy_generation():
    """Test both rule-based and OpenAI strategy generation"""
    print("🧠 Testing Enhanced Strategy Generation")
    print("=" * 50)
    
    engine = init_db_and_create('sqlite:///./finbrief_full.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Test 1: Rule-based generation (no API key)
    print("\n📊 Test 1: Rule-based Strategy Generation")
    print("-" * 30)
    generator_rules = StrategyGenerator()
    
    strategy_rules = generator_rules.create_strategy(session, StrategyHorizonEnum.weekly, MarketEnum.global_market)
    
    if strategy_rules:
        print(f"✅ Title: {strategy_rules.title}")
        print(f"📝 Summary: {strategy_rules.summary}")
        print(f"🎯 Confidence: {strategy_rules.confidence_score}")
        print(f"📈 Key Drivers: {strategy_rules.key_drivers}")
    
    # Test 2: OpenAI generation (if API key provided)
    print("\n🤖 Test 2: OpenAI Strategy Generation")
    print("-" * 30)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"OpenAI API key found: {openai_key[:12]}...")
        generator_ai = StrategyGenerator(openai_api_key=openai_key)
        
        strategy_ai = generator_ai.create_strategy(session, StrategyHorizonEnum.daily, MarketEnum.global_market)
        
        if strategy_ai:
            print(f"✅ AI Title: {strategy_ai.title}")
            print(f"📝 AI Summary: {strategy_ai.summary}")
            print(f"🎯 AI Confidence: {strategy_ai.confidence_score}")
            print(f"📈 AI Key Drivers: {strategy_ai.key_drivers}")
            print(f"💡 AI Recommendations: {strategy_ai.action_recommendations}")
        else:
            print("❌ Failed to generate AI strategy")
    else:
        print("⚠️  No OPENAI_API_KEY environment variable set")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        print("   Then run: python test_openai_strategy.py")
    
    session.close()
    
    print("\n🎉 Strategy generation testing complete!")
    print("\nTo test with OpenAI:")
    print("1. Get API key from: https://platform.openai.com/api-keys")
    print("2. export OPENAI_API_KEY='your-key-here'")
    print("3. python test_openai_strategy.py")

if __name__ == "__main__":
    test_strategy_generation()