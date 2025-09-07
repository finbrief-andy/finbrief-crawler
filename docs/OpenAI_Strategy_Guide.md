# OpenAI-Powered Strategy Generation

## Overview

FinBrief now supports intelligent strategy generation using OpenAI's GPT-4 models, with automatic fallback to rule-based generation when OpenAI is not available.

## Features

### 🤖 **AI-Powered Analysis**
- Uses GPT-4o-mini for cost-effective, high-quality strategy generation
- Analyzes real financial news with context-aware prompts
- Generates actionable investment recommendations across multiple time horizons

### 🔄 **Automatic Fallback**
- Falls back to rule-based generation if OpenAI is unavailable
- Ensures system reliability and continuous operation
- No breaking changes to existing functionality

### 🎯 **Multi-Horizon Strategy**
- **Daily**: Short-term trading decisions (24 hours)
- **Weekly**: Portfolio positioning (1 week)  
- **Monthly**: Sector allocation (1 month)
- **Yearly**: Long-term investment outlook (1 year)

## Setup

### 1. Install Dependencies
```bash
pip install openai
```

### 2. Get OpenAI API Key
1. Visit https://platform.openai.com/api-keys
2. Create a new API key
3. Set environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Test Installation
```bash
python test_openai_strategy.py
```

## Usage

### Basic Usage
```python
from src.services.strategy_generator import StrategyGenerator
from src.database.models_migration import StrategyHorizonEnum, MarketEnum

# Initialize with OpenAI
generator = StrategyGenerator(openai_api_key="your-key")

# Generate daily strategy for global market
strategy = generator.create_strategy(session, StrategyHorizonEnum.daily, MarketEnum.global_market)
```

### API Integration
The strategy generation is automatically used when calling:
```bash
# Generate new strategies via API
curl -X POST http://localhost:8000/strategy/generate \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get latest strategy
curl http://localhost:8000/strategy/daily?market=global
```

## Strategy Output Format

OpenAI generates strategies in this structured format:

```json
{
  "title": "Weekly Market Recovery Strategy",
  "summary": "Mixed signals suggest cautious optimism with selective buying opportunities in tech sector.",
  "key_drivers": [
    "Tech earnings showing resilience despite market volatility",
    "Federal Reserve policy signals suggesting rate stability",
    "Global supply chain improvements in key sectors"
  ],
  "action_recommendations": [
    {
      "action": "BUY",
      "asset_focus": "stocks",
      "rationale": "Technology sector showing strong fundamentals with positive earnings momentum",
      "confidence": 0.75,
      "time_sensitivity": "medium"
    }
  ],
  "confidence_score": 0.78
}
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required for AI generation)
- `OPENAI_MODEL`: Model to use (default: "gpt-4o-mini")

### Model Selection
Current setup uses `gpt-4o-mini` for optimal cost/performance balance:
- **Cost**: ~10x cheaper than GPT-4
- **Quality**: Near GPT-4 performance for financial analysis
- **Speed**: Fast response times

## Cost Optimization

### Token Usage
- **Input**: ~300-500 tokens per strategy (news context + prompt)
- **Output**: ~200-400 tokens per strategy
- **Cost**: ~$0.01-0.02 per strategy generation

### Best Practices
1. **Batch generation**: Generate multiple horizons together
2. **Caching**: Store strategies for reuse within time windows
3. **Fallback**: Always available rule-based backup

## Monitoring

### Logs
The system logs all OpenAI interactions:
```python
# Success
INFO: Generated OpenAI strategy for daily horizon

# Fallback 
ERROR: OpenAI API error: Rate limit exceeded. Falling back to rule-based generation
```

### Error Handling
- **API Errors**: Automatic fallback to rule-based generation
- **JSON Parsing**: Retry with error correction
- **Rate Limits**: Graceful degradation with logging

## Testing

### Test Without API Key (Rule-based)
```bash
python test_openai_strategy.py
```

### Test With API Key (AI-powered)
```bash
export OPENAI_API_KEY='your-key'
python test_openai_strategy.py
```

## Production Deployment

### Required Environment Variables
```bash
export OPENAI_API_KEY='your-production-key'
export SECRET_KEY='your-app-secret'
export DATABASE_URI='postgresql://...'
```

### Scaling Considerations
- **Rate Limits**: OpenAI has rate limits per minute/day
- **Cost Control**: Monitor token usage in production
- **Reliability**: Rule-based fallback ensures uptime

## Examples

### Vietnamese Market Strategy
```python
# Generate VN market strategy  
strategy = generator.create_strategy(
    session, 
    StrategyHorizonEnum.weekly, 
    MarketEnum.vn
)
```

### Batch Strategy Generation
```python
# Generate all horizons at once
all_strategies = generator.generate_all_strategies(session, MarketEnum.global_market)
```

## Troubleshooting

### Common Issues

1. **"No OpenAI API key provided"**
   - Set `OPENAI_API_KEY` environment variable
   - Check key validity at OpenAI dashboard

2. **"Rate limit exceeded"**
   - Wait and retry (automatic fallback active)
   - Consider upgrading OpenAI plan

3. **"Failed to parse OpenAI JSON response"**
   - System automatically retries with rule-based fallback
   - Check logs for specific parsing errors

4. **High costs**
   - Monitor token usage in OpenAI dashboard
   - Consider reducing strategy generation frequency
   - Use batch generation for multiple horizons

## Next Steps

1. **Custom Prompts**: Customize prompts for specific markets/sectors
2. **Fine-tuning**: Train custom models on financial data
3. **Multi-model**: Compare strategies from different AI providers
4. **Personalization**: User-specific strategy customization

---

🚀 **Your FinBrief system now has AI-powered investment intelligence!**