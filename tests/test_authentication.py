#!/usr/bin/env python3
"""
Comprehensive authentication testing for FinBrief API.
Tests JWT validation, user flows, role-based access, and endpoint security.
"""
import sys
import os
import pytest
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jwt
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models_migration import Base, User, RoleEnum, init_db_and_create
from scripts.main import app, get_db, SECRET_KEY, ALGORITHM, hash_password


# Test fixtures
@pytest.fixture
def test_db():
    """Create test database"""
    # Use in-memory SQLite for tests
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal
    
    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_db):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def test_users(test_db):
    """Create test users with different roles"""
    db = test_db()
    
    users = {
        'admin': User(
            email="admin@test.com",
            password_hash=hash_password("admin123"),
            role=RoleEnum.admin
        ),
        'analyst': User(
            email="analyst@test.com", 
            password_hash=hash_password("analyst123"),
            role=RoleEnum.analyst
        ),
        'user': User(
            email="user@test.com",
            password_hash=hash_password("user123"), 
            role=RoleEnum.user
        )
    }
    
    for user in users.values():
        db.add(user)
    
    db.commit()
    
    for user in users.values():
        db.refresh(user)
    
    db.close()
    return users


class TestJWTTokenValidation:
    """Test JWT token creation and validation"""
    
    def test_valid_token_creation(self):
        """Test creating valid JWT tokens"""
        from scripts.main import create_access_token
        
        # Test token creation
        token = create_access_token({"sub": "123"})
        assert token is not None
        
        # Decode and verify
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "123"
        assert "exp" in payload
    
    def test_token_expiration(self):
        """Test token expiration handling"""
        from scripts.main import create_access_token
        
        # Create expired token
        expired_token = create_access_token(
            {"sub": "123"},
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        
        # Should raise exception when decoding
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(expired_token, SECRET_KEY, algorithms=[ALGORITHM])
    
    def test_invalid_token_signature(self):
        """Test invalid token signatures"""
        # Create token with wrong key
        invalid_token = jwt.encode({"sub": "123"}, "wrong-key", algorithm=ALGORITHM)
        
        with pytest.raises(jwt.InvalidSignatureError):
            jwt.decode(invalid_token, SECRET_KEY, algorithms=[ALGORITHM])
    
    def test_malformed_tokens(self):
        """Test malformed token handling"""
        malformed_tokens = [
            "invalid.token",
            "not.a.token.at.all",
            "",
            "Bearer invalid-token"
        ]
        
        for token in malformed_tokens:
            with pytest.raises(jwt.DecodeError):
                jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


class TestUserRegistrationAndLogin:
    """Test user registration and login flows"""
    
    def test_successful_registration(self, client):
        """Test successful user registration"""
        response = client.post("/auth/signup", json={
            "email": "newuser@test.com",
            "password": "password123"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newuser@test.com"
        assert "id" in data
        assert "created_at" in data
        assert "password" not in data  # Should not return password
    
    def test_duplicate_email_registration(self, client, test_users):
        """Test registration with existing email"""
        response = client.post("/auth/signup", json={
            "email": "admin@test.com",  # Already exists
            "password": "password123"
        })
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()
    
    def test_successful_login(self, client, test_users):
        """Test successful login"""
        response = client.post("/auth/login", data={
            "username": "admin@test.com",
            "password": "admin123"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["token_type"] == "bearer"
        assert "access_token" in data
        
        # Verify token is valid
        token = data["access_token"]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == str(test_users["admin"].id)
    
    def test_invalid_credentials(self, client, test_users):
        """Test login with invalid credentials"""
        # Wrong password
        response = client.post("/auth/login", data={
            "username": "admin@test.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401
        
        # Non-existent user
        response = client.post("/auth/login", data={
            "username": "nonexistent@test.com",
            "password": "password123"
        })
        assert response.status_code == 401
    
    def test_get_current_user(self, client, test_users):
        """Test getting current user info"""
        # Login first
        login_response = client.post("/auth/login", data={
            "username": "admin@test.com",
            "password": "admin123"
        })
        token = login_response.json()["access_token"]
        
        # Get user info
        response = client.get("/auth/me", headers={
            "Authorization": f"Bearer {token}"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "admin@test.com"
        assert data["id"] == test_users["admin"].id


class TestRoleBasedAccessControl:
    """Test role-based access control"""
    
    def get_auth_headers(self, client, email: str, password: str) -> Dict[str, str]:
        """Helper to get authorization headers"""
        response = client.post("/auth/login", data={
            "username": email,
            "password": password
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_admin_strategy_generation_access(self, client, test_users):
        """Test admin can generate strategies"""
        headers = self.get_auth_headers(client, "admin@test.com", "admin123")
        
        response = client.post("/strategy/generate", 
                             params={"market": "global", "horizons": ["daily"]},
                             headers=headers)
        
        # Should succeed (admin access)
        assert response.status_code == 200
    
    def test_user_strategy_generation_denied(self, client, test_users):
        """Test regular user cannot generate strategies"""
        headers = self.get_auth_headers(client, "user@test.com", "user123")
        
        response = client.post("/strategy/generate",
                             params={"market": "global", "horizons": ["daily"]}, 
                             headers=headers)
        
        # Should be denied (403 Forbidden)
        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()
    
    def test_analyst_permissions(self, client, test_users):
        """Test analyst role permissions"""
        headers = self.get_auth_headers(client, "analyst@test.com", "analyst123")
        
        # Analyst should be denied admin functions  
        response = client.post("/strategy/generate",
                             params={"market": "global"},
                             headers=headers)
        assert response.status_code == 403
    
    def test_unauthenticated_access_denied(self, client):
        """Test unauthenticated users are denied access"""
        # Try to access protected endpoint without token
        response = client.get("/auth/me")
        assert response.status_code == 401
        
        response = client.post("/feedback", json={
            "analysis_id": 1,
            "vote": "agree"
        })
        assert response.status_code == 401


class TestAPIEndpointSecurity:
    """Test API endpoint security"""
    
    def get_valid_token(self, client, email: str, password: str) -> str:
        """Helper to get valid token"""
        response = client.post("/auth/login", data={
            "username": email, 
            "password": password
        })
        return response.json()["access_token"]
    
    def test_protected_endpoints_require_auth(self, client, test_users):
        """Test that protected endpoints require authentication"""
        protected_endpoints = [
            ("GET", "/auth/me"),
            ("POST", "/feedback"),
            ("POST", "/strategy/generate"),
        ]
        
        for method, endpoint in protected_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json={})
            
            assert response.status_code == 401, f"{method} {endpoint} should require auth"
    
    def test_invalid_token_handling(self, client):
        """Test handling of invalid tokens"""
        invalid_tokens = [
            "invalid-token",
            "Bearer invalid-token",
            "Bearer ",
            "",
        ]
        
        for token in invalid_tokens:
            response = client.get("/auth/me", headers={
                "Authorization": token
            })
            assert response.status_code == 401
    
    def test_expired_token_handling(self, client, test_users):
        """Test expired token handling"""
        from scripts.main import create_access_token
        
        # Create expired token
        expired_token = create_access_token(
            {"sub": str(test_users["user"].id)},
            expires_delta=timedelta(seconds=-1)
        )
        
        response = client.get("/auth/me", headers={
            "Authorization": f"Bearer {expired_token}"
        })
        assert response.status_code == 401
    
    def test_token_with_invalid_user_id(self, client):
        """Test token with non-existent user ID"""
        from scripts.main import create_access_token
        
        # Create token with non-existent user ID
        token = create_access_token({"sub": "999999"})
        
        response = client.get("/auth/me", headers={
            "Authorization": f"Bearer {token}"
        })
        assert response.status_code == 404  # User not found
    
    def test_sql_injection_prevention(self, client):
        """Test SQL injection prevention in auth endpoints"""
        # Try SQL injection in login
        response = client.post("/auth/login", data={
            "username": "admin@test.com'; DROP TABLE users; --",
            "password": "anything"
        })
        # Should fail safely, not crash
        assert response.status_code in [401, 422]
        
        # Try SQL injection in signup
        response = client.post("/auth/signup", json={
            "email": "test'; DROP TABLE users; --@test.com",
            "password": "password123"
        })
        # Should handle gracefully
        assert response.status_code in [400, 422]
    
    def test_password_security(self, client):
        """Test password security measures"""
        # Test that passwords are hashed
        response = client.post("/auth/signup", json={
            "email": "testpassword@test.com",
            "password": "mypassword123"
        })
        assert response.status_code == 200
        
        # Password should not appear in response
        data = response.json()
        assert "password" not in data
        assert "password_hash" not in data
    
    def test_email_validation(self, client):
        """Test email validation"""
        invalid_emails = [
            "invalid-email",
            "@test.com",
            "test@",
            "",
            "spaces @test.com"
        ]
        
        for email in invalid_emails:
            response = client.post("/auth/signup", json={
                "email": email,
                "password": "password123"
            })
            # Should reject invalid emails
            assert response.status_code == 422


class TestComprehensiveAuthFlow:
    """Test complete authentication flows"""
    
    def test_complete_user_journey(self, client):
        """Test complete user registration to protected resource access"""
        # 1. Register new user
        signup_response = client.post("/auth/signup", json={
            "email": "journey@test.com",
            "password": "journey123"
        })
        assert signup_response.status_code == 200
        user_data = signup_response.json()
        
        # 2. Login with new user
        login_response = client.post("/auth/login", data={
            "username": "journey@test.com",
            "password": "journey123"
        })
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        
        # 3. Access protected resource
        me_response = client.get("/auth/me", headers={
            "Authorization": f"Bearer {token}"
        })
        assert me_response.status_code == 200
        me_data = me_response.json()
        assert me_data["id"] == user_data["id"]
        assert me_data["email"] == "journey@test.com"
        
        # 4. Try to submit feedback (should work for regular user)
        feedback_response = client.post("/feedback", 
                                      json={
                                          "analysis_id": 1,
                                          "vote": "agree"
                                      },
                                      headers={"Authorization": f"Bearer {token}"})
        # May fail due to missing analysis, but should not be auth error
        assert feedback_response.status_code != 401
    
    def test_token_refresh_behavior(self, client, test_users):
        """Test token behavior over time"""
        # Get initial token
        token1 = self.get_token(client, "user@test.com", "user123")
        
        # Use token
        response1 = client.get("/auth/me", headers={
            "Authorization": f"Bearer {token1}"
        })
        assert response1.status_code == 200
        
        # Get new token (simulate refresh)
        token2 = self.get_token(client, "user@test.com", "user123")
        
        # Both tokens should work (until expiration)
        response2 = client.get("/auth/me", headers={
            "Authorization": f"Bearer {token2}"
        })
        assert response2.status_code == 200
    
    def get_token(self, client, email: str, password: str) -> str:
        """Helper to get authentication token"""
        response = client.post("/auth/login", data={
            "username": email,
            "password": password
        })
        return response.json()["access_token"]


# Integration test runner
def run_auth_tests():
    """Run authentication tests manually"""
    print("🔐 Running Authentication Tests")
    print("=" * 40)
    
    # This would normally be run with pytest, but we can run some basic checks
    try:
        # Test JWT functions
        from scripts.main import create_access_token, hash_password, verify_password
        
        # Test password hashing
        password = "test123"
        hashed = hash_password(password)
        assert verify_password(password, hashed)
        assert not verify_password("wrong", hashed)
        print("✅ Password hashing works")
        
        # Test token creation
        token = create_access_token({"sub": "123"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "123"
        print("✅ JWT token creation works")
        
        print("\n✅ Basic auth functions working!")
        print("Run full tests with: pytest tests/test_authentication.py -v")
        
    except Exception as e:
        print(f"❌ Auth test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    run_auth_tests()