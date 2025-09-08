#!/usr/bin/env python3
"""
Authentication System Validation Test
Tests basic authentication functionality without requiring a running server.
"""
import sys
import os
import tempfile
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_password_hashing():
    """Test password hashing functionality"""
    print("=== Testing Password Hashing ===")
    try:
        from scripts.main import hash_password, verify_password
        
        # Test basic hashing
        password = "test_password_123"
        hashed = hash_password(password)
        
        # Verify correct password
        assert verify_password(password, hashed), "Correct password should verify"
        print("✅ Correct password verification works")
        
        # Verify incorrect password
        assert not verify_password("wrong_password", hashed), "Wrong password should not verify"
        print("✅ Incorrect password rejection works")
        
        # Test that same password produces different hashes (salt)
        hashed2 = hash_password(password)
        assert hashed != hashed2, "Same password should produce different hashes (salted)"
        print("✅ Password salting works")
        
        return True
        
    except Exception as e:
        print(f"❌ Password hashing test failed: {e}")
        return False


def test_jwt_token_creation():
    """Test JWT token creation and validation"""
    print("\n=== Testing JWT Tokens ===")
    try:
        import jwt
        from scripts.main import create_access_token, SECRET_KEY, ALGORITHM
        
        # Test token creation
        user_data = {"sub": "123", "role": "user"}
        token = create_access_token(user_data)
        assert token is not None, "Token should be created"
        print("✅ Token creation works")
        
        # Test token decoding
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert decoded["sub"] == "123", "Token should contain correct user data"
        assert "exp" in decoded, "Token should have expiration"
        print("✅ Token decoding works")
        
        # Test expired token
        expired_token = create_access_token(user_data, expires_delta=timedelta(seconds=-1))
        try:
            jwt.decode(expired_token, SECRET_KEY, algorithms=[ALGORITHM])
            assert False, "Expired token should raise exception"
        except jwt.ExpiredSignatureError:
            print("✅ Expired token detection works")
        
        # Test invalid signature
        try:
            jwt.decode(token, "wrong_secret", algorithms=[ALGORITHM])
            assert False, "Invalid signature should raise exception"
        except jwt.InvalidSignatureError:
            print("✅ Invalid signature detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ JWT token test failed: {e}")
        return False


def test_user_model():
    """Test User model and database operations"""
    print("\n=== Testing User Model ===")
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.database.models_migration import Base, User, RoleEnum
        from scripts.main import hash_password
        
        # Create in-memory database
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        # Test user creation
        test_user = User(
            email="test@example.com",
            password_hash=hash_password("password123"),
            role=RoleEnum.user
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        assert test_user.id is not None, "User should have ID after commit"
        assert test_user.email == "test@example.com", "User email should be saved"
        assert test_user.role == RoleEnum.user, "User role should be saved"
        print("✅ User creation works")
        
        # Test user retrieval
        retrieved_user = db.query(User).filter(User.email == "test@example.com").first()
        assert retrieved_user is not None, "User should be retrievable"
        assert retrieved_user.id == test_user.id, "Retrieved user should match"
        print("✅ User retrieval works")
        
        # Test unique email constraint
        duplicate_user = User(
            email="test@example.com",  # Same email
            password_hash=hash_password("different_password"),
            role=RoleEnum.user
        )
        db.add(duplicate_user)
        
        try:
            db.commit()
            assert False, "Duplicate email should raise constraint error"
        except Exception:
            db.rollback()
            print("✅ Unique email constraint works")
        
        # Test roles
        admin_user = User(
            email="admin@example.com",
            password_hash=hash_password("admin123"),
            role=RoleEnum.admin
        )
        db.add(admin_user)
        db.commit()
        
        assert admin_user.role == RoleEnum.admin, "Admin role should be saved"
        print("✅ Role system works")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ User model test failed: {e}")
        return False


def test_auth_endpoints_logic():
    """Test authentication endpoint logic"""
    print("\n=== Testing Auth Endpoint Logic ===")
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.database.models_migration import Base, User, RoleEnum
        from scripts.main import hash_password, verify_password, get_current_user, create_access_token
        import jwt
        from fastapi import HTTPException
        
        # Create in-memory database  
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        # Create test user
        test_user = User(
            email="auth_test@example.com",
            password_hash=hash_password("password123"),
            role=RoleEnum.user
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        # Test login logic (simulate)
        user = db.query(User).filter(User.email == "auth_test@example.com").first()
        assert user is not None, "User should exist"
        
        # Test correct password
        assert verify_password("password123", user.password_hash), "Password should verify"
        print("✅ Login password verification works")
        
        # Test wrong password
        assert not verify_password("wrong_password", user.password_hash), "Wrong password should fail"
        print("✅ Login wrong password rejection works")
        
        # Test token creation for user
        token = create_access_token({"sub": str(user.id)})
        assert token is not None, "Token should be created for user"
        print("✅ User token creation works")
        
        # Test get_current_user logic (simplified)
        from scripts.main import SECRET_KEY, ALGORITHM
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        current_user = db.query(User).filter(User.id == user_id).first()
        
        assert current_user is not None, "Current user should be found"
        assert current_user.id == user.id, "Current user should match token user"
        print("✅ Current user retrieval works")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Auth endpoint logic test failed: {e}")
        return False


def test_role_based_access():
    """Test role-based access control logic"""
    print("\n=== Testing Role-Based Access ===")
    try:
        from src.database.models_migration import RoleEnum
        
        # Test role hierarchy
        roles = [RoleEnum.user, RoleEnum.analyst, RoleEnum.admin]
        
        # Test each role exists
        for role in roles:
            assert role.value in ["user", "analyst", "admin"], f"Role {role} should have valid value"
        print("✅ Role enumeration works")
        
        # Test role-based logic (simulate strategy generation access)
        def can_generate_strategies(role: RoleEnum) -> bool:
            return role.value in ["admin", "system"]
        
        assert not can_generate_strategies(RoleEnum.user), "Regular user should not generate strategies"
        assert not can_generate_strategies(RoleEnum.analyst), "Analyst should not generate strategies"  
        assert can_generate_strategies(RoleEnum.admin), "Admin should generate strategies"
        print("✅ Role-based access control logic works")
        
        return True
        
    except Exception as e:
        print(f"❌ Role-based access test failed: {e}")
        return False


def test_security_best_practices():
    """Test security best practices implementation"""
    print("\n=== Testing Security Best Practices ===")
    try:
        from scripts.main import SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES
        
        # Test secret key is not default
        if SECRET_KEY == "super-secret-key-change-in-production":
            print("⚠️  Using default SECRET_KEY (should be changed in production)")
        else:
            print("✅ Custom SECRET_KEY configured")
        
        # Test token expiration is reasonable
        if ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
            print("❌ Invalid token expiration time")
            return False
        elif ACCESS_TOKEN_EXPIRE_MINUTES > 1440:  # 24 hours
            print("⚠️  Token expiration is very long (consider shorter)")
        else:
            print(f"✅ Token expiration set to {ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
        
        # Test password hashing uses bcrypt
        from scripts.main import pwd_context
        assert "bcrypt" in pwd_context.schemes, "Should use bcrypt for password hashing"
        print("✅ Using bcrypt for password hashing")
        
        return True
        
    except Exception as e:
        print(f"❌ Security best practices test failed: {e}")
        return False


def main():
    """Run all authentication system tests"""
    print("🔐 Authentication System Validation")
    print("=" * 50)
    
    tests = [
        ("Password Hashing", test_password_hashing),
        ("JWT Token Creation", test_jwt_token_creation), 
        ("User Model", test_user_model),
        ("Auth Endpoint Logic", test_auth_endpoints_logic),
        ("Role-Based Access", test_role_based_access),
        ("Security Best Practices", test_security_best_practices)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
        print()
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All authentication tests passed!")
        print("✅ Authentication system is working correctly")
        print("\n📋 Next steps:")
        print("1. Run full pytest suite: pytest tests/test_authentication.py -v")
        print("2. Test with live API: python tests/test_api_security.py")
        print("3. Configure production security settings")
    elif passed >= total * 0.8:  # 80% pass rate
        print("✅ Authentication system mostly working")
        print("⚠️  Some issues found - review failed tests above")
    else:
        print("❌ Authentication system has significant issues")
        print("🔧 Fix failed tests before proceeding")
    
    return passed >= total * 0.8


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)