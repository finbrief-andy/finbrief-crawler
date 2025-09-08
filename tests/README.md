# FinBrief Test Suite

This directory contains organized test files for the FinBrief financial news crawler system.

## Directory Structure

### `/api` - API Endpoint Tests
- `test_backend.py` - Backend API endpoint testing

### `/auth` - Authentication & Security Tests  
- `test_authentication.py` - Comprehensive authentication test suite (6 test classes)
- `test_api_security.py` - Live API security testing utility
- `test_auth_basic.py` - Basic authentication validation tests
- `test_auth_system.py` - Authentication system integration tests

### `/crawlers` - News Crawler Tests
- `test_new_sources.py` - New news source integration tests
- `test_new_adapters_only.py` - Individual adapter testing

### `/database` - Database Tests
- `test_production_db.py` - PostgreSQL production database tests

### `/integration` - Integration & System Tests
- `test_pipeline.py` - Full news processing pipeline tests
- `test_scheduler.py` - Automated scheduler system tests
- `test_scheduler_quick.py` - Quick scheduler validation

### `/services` - Service Layer Tests
- `test_openai_strategy.py` - OpenAI strategy generation tests
- `test_enhanced_nlp.py` - Advanced NLP processing tests
- `test_vector_search.py` - Vector search and semantic analysis tests

### `/unit` - Unit Tests
- `test_nlp_basic.py` - Basic NLP functionality tests
- `test_nlp_adapter.py` - NLP adapter compatibility tests

## Running Tests

### Run All Tests
```bash
python -m pytest tests/
```

### Run Tests by Category
```bash
# API tests
python -m pytest tests/api/

# Authentication tests
python -m pytest tests/auth/

# Service tests
python -m pytest tests/services/

# Integration tests
python -m pytest tests/integration/
```

### Run Individual Test Files
```bash
# Example: Run vector search tests
python tests/services/test_vector_search.py

# Example: Run authentication tests  
python tests/auth/test_authentication.py
```

## Test Categories

- **Unit Tests**: Test individual functions and components in isolation
- **Integration Tests**: Test interactions between multiple components
- **API Tests**: Test REST API endpoints and responses
- **Service Tests**: Test business logic and service layer functionality
- **Auth Tests**: Test authentication, authorization, and security features
- **Database Tests**: Test database operations and data integrity

## Test Dependencies

Some tests require specific dependencies:
- ChromaDB, sentence-transformers, scikit-learn (for vector search tests)
- OpenAI API key (for strategy generation tests)
- PostgreSQL database connection (for database tests)

## Development Notes

- Tests are organized by functionality rather than file location
- Each test file is self-contained and can be run independently
- Integration tests may require database setup and external dependencies
- All test files follow the `test_*.py` naming convention for pytest discovery