#!/usr/bin/env python3
"""
Debug JWT token creation and validation
"""
import jwt
import os
from datetime import datetime, timedelta
from src.database.models_migration import init_db_and_create, User
from sqlalchemy.orm import sessionmaker

# Same settings as main.py
SECRET_KEY = "finbrief-super-secret-key-12345"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def validate_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("✅ Token decoded successfully:")
        print(f"  Payload: {payload}")
        user_id = payload.get("sub")
        print(f"  User ID: {user_id}")
        return payload
    except jwt.ExpiredSignatureError:
        print("❌ Token expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid token: {e}")
        return None

if __name__ == "__main__":
    print("🔧 Debugging JWT Token System")
    print(f"SECRET_KEY: {SECRET_KEY}")
    print(f"ALGORITHM: {ALGORITHM}")
    
    # Create a test token
    test_data = {"sub": "1"}
    print(f"Creating token with data: {test_data}")
    
    token = create_access_token(test_data)
    print(f"Generated token: {token}")
    
    # Validate the token
    print("\\nValidating token...")
    result = validate_token(token)
    
    if result:
        print("\\n✅ JWT system working correctly!")
    else:
        print("\\n❌ JWT system has issues!")