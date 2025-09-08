#!/usr/bin/env python3
"""
API Security Testing for FinBrief.
Tests for common security vulnerabilities and best practices.
"""
import sys
import os
import requests
import time
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class APISecurityTester:
    """Security testing for FinBrief API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   {details}")
        
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
    
    def test_cors_headers(self):
        """Test CORS configuration"""
        try:
            # Test preflight request
            response = self.session.options(f"{self.base_url}/auth/login")
            
            # Check for proper CORS headers
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers'
            ]
            
            has_cors = any(header in response.headers for header in cors_headers)
            
            if response.status_code == 405:
                # OPTIONS not allowed - this might be intentional
                self.log_test("CORS Headers", True, "OPTIONS method disabled (security best practice)")
            elif has_cors:
                self.log_test("CORS Headers", True, "CORS headers present")
            else:
                self.log_test("CORS Headers", False, "No CORS headers found")
                
        except Exception as e:
            self.log_test("CORS Headers", False, f"Error: {e}")
    
    def test_security_headers(self):
        """Test for security headers"""
        try:
            response = self.session.get(f"{self.base_url}/")
            
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
            }
            
            present_headers = []
            missing_headers = []
            
            for header, expected_value in security_headers.items():
                if header in response.headers:
                    present_headers.append(header)
                else:
                    missing_headers.append(header)
            
            if len(present_headers) > 0:
                self.log_test("Security Headers", True, f"Present: {', '.join(present_headers)}")
            
            if len(missing_headers) > 0:
                self.log_test("Missing Security Headers", False, f"Missing: {', '.join(missing_headers)}")
            
        except Exception as e:
            self.log_test("Security Headers", False, f"Error: {e}")
    
    def test_rate_limiting(self):
        """Test for rate limiting"""
        try:
            # Make rapid requests to login endpoint
            responses = []
            for i in range(10):
                response = self.session.post(f"{self.base_url}/auth/login", 
                                           data={"username": "test", "password": "test"})
                responses.append(response.status_code)
                time.sleep(0.1)
            
            # Check if any requests were rate limited (429)
            rate_limited = any(status == 429 for status in responses)
            
            if rate_limited:
                self.log_test("Rate Limiting", True, "Rate limiting detected")
            else:
                self.log_test("Rate Limiting", False, "No rate limiting detected (consider implementing)")
                
        except Exception as e:
            self.log_test("Rate Limiting", False, f"Error: {e}")
    
    def test_sql_injection_protection(self):
        """Test SQL injection protection"""
        try:
            # Common SQL injection payloads
            sql_payloads = [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "' UNION SELECT * FROM users --",
                "admin'--",
                "' OR 1=1#"
            ]
            
            injection_blocked = True
            
            for payload in sql_payloads:
                response = self.session.post(f"{self.base_url}/auth/login", 
                                           data={"username": payload, "password": "test"})
                
                # Should return 401 (unauthorized) or 422 (validation error), not 500 (server error)
                if response.status_code == 500:
                    injection_blocked = False
                    break
            
            self.log_test("SQL Injection Protection", injection_blocked, 
                         "SQL injection payloads handled safely" if injection_blocked else "Potential SQL injection vulnerability")
                         
        except Exception as e:
            self.log_test("SQL Injection Protection", False, f"Error: {e}")
    
    def test_xss_protection(self):
        """Test XSS protection"""
        try:
            # Common XSS payloads
            xss_payloads = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>",
                "'><script>alert('xss')</script>"
            ]
            
            xss_blocked = True
            
            for payload in xss_payloads:
                response = self.session.post(f"{self.base_url}/auth/signup", 
                                           json={"email": payload, "password": "test123"})
                
                # Check if payload is reflected in response without proper encoding
                if payload in response.text and "<script>" in response.text:
                    xss_blocked = False
                    break
            
            self.log_test("XSS Protection", xss_blocked,
                         "XSS payloads handled safely" if xss_blocked else "Potential XSS vulnerability")
                         
        except Exception as e:
            self.log_test("XSS Protection", False, f"Error: {e}")
    
    def test_information_disclosure(self):
        """Test for information disclosure"""
        try:
            # Test error responses don't reveal sensitive info
            response = self.session.get(f"{self.base_url}/nonexistent-endpoint")
            
            sensitive_keywords = [
                "traceback",
                "sqlalchemy",
                "database",
                "password",
                "secret_key",
                "internal server error"
            ]
            
            disclosed_info = []
            for keyword in sensitive_keywords:
                if keyword.lower() in response.text.lower():
                    disclosed_info.append(keyword)
            
            if len(disclosed_info) == 0:
                self.log_test("Information Disclosure", True, "No sensitive information in error responses")
            else:
                self.log_test("Information Disclosure", False, f"Potential info disclosure: {', '.join(disclosed_info)}")
                
        except Exception as e:
            self.log_test("Information Disclosure", False, f"Error: {e}")
    
    def test_authentication_bypass(self):
        """Test for authentication bypass vulnerabilities"""
        try:
            # Test accessing protected endpoints without authentication
            protected_endpoints = [
                "/auth/me",
                "/feedback",
                "/strategy/generate"
            ]
            
            bypass_detected = False
            
            for endpoint in protected_endpoints:
                # Try without authentication
                response = self.session.get(f"{self.base_url}{endpoint}")
                
                # Should return 401 (unauthorized)
                if response.status_code != 401:
                    bypass_detected = True
                    break
            
            self.log_test("Authentication Bypass", not bypass_detected,
                         "Protected endpoints properly secured" if not bypass_detected else "Potential auth bypass detected")
                         
        except Exception as e:
            self.log_test("Authentication Bypass", False, f"Error: {e}")
    
    def test_jwt_security(self):
        """Test JWT token security"""
        try:
            # Try to create account and get token
            signup_response = self.session.post(f"{self.base_url}/auth/signup", 
                                              json={"email": "security@test.com", "password": "test123"})
            
            if signup_response.status_code != 200:
                self.log_test("JWT Security", False, "Could not create test account")
                return
            
            login_response = self.session.post(f"{self.base_url}/auth/login",
                                             data={"username": "security@test.com", "password": "test123"})
            
            if login_response.status_code != 200:
                self.log_test("JWT Security", False, "Could not login")
                return
            
            token = login_response.json().get("access_token")
            
            if not token:
                self.log_test("JWT Security", False, "No token returned")
                return
            
            # Test token structure
            token_parts = token.split(".")
            jwt_secure = len(token_parts) == 3  # Header.Payload.Signature
            
            # Test with modified token (should fail)
            modified_token = token[:-5] + "aaaaa"  # Change last 5 chars
            response = self.session.get(f"{self.base_url}/auth/me",
                                      headers={"Authorization": f"Bearer {modified_token}"})
            
            token_validation = response.status_code == 401  # Should reject modified token
            
            self.log_test("JWT Security", jwt_secure and token_validation,
                         "JWT tokens properly structured and validated" if jwt_secure and token_validation 
                         else "JWT security issues detected")
                         
        except Exception as e:
            self.log_test("JWT Security", False, f"Error: {e}")
    
    def test_input_validation(self):
        """Test input validation"""
        try:
            # Test with various invalid inputs
            invalid_inputs = [
                {"email": "", "password": "test123"},  # Empty email
                {"email": "test", "password": ""},     # Empty password
                {"email": "a" * 1000, "password": "test123"},  # Very long email
                {"email": "test@test.com", "password": "a" * 1000},  # Very long password
                {"email": None, "password": "test123"},  # Null values
                {"email": "test@test.com", "password": None}
            ]
            
            validation_working = True
            
            for invalid_input in invalid_inputs:
                response = self.session.post(f"{self.base_url}/auth/signup", json=invalid_input)
                
                # Should return 422 (validation error) or 400 (bad request), not 500 (server error)
                if response.status_code == 500:
                    validation_working = False
                    break
            
            self.log_test("Input Validation", validation_working,
                         "Input validation working properly" if validation_working 
                         else "Input validation issues detected")
                         
        except Exception as e:
            self.log_test("Input Validation", False, f"Error: {e}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests"""
        print("🔒 Running API Security Tests")
        print("=" * 40)
        
        # Check if API is running
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code != 200:
                print(f"❌ API not accessible at {self.base_url}")
                return {"error": "API not accessible"}
        except Exception as e:
            print(f"❌ Cannot connect to API at {self.base_url}: {e}")
            return {"error": f"Cannot connect: {e}"}
        
        print(f"✅ API accessible at {self.base_url}")
        print()
        
        # Run all tests
        tests = [
            self.test_cors_headers,
            self.test_security_headers,
            self.test_rate_limiting,
            self.test_sql_injection_protection,
            self.test_xss_protection,
            self.test_information_disclosure,
            self.test_authentication_bypass,
            self.test_jwt_security,
            self.test_input_validation
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.log_test(test.__name__.replace('test_', '').replace('_', ' ').title(), 
                             False, f"Test error: {e}")
            print()
        
        # Summary
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        total_tests = len(self.test_results)
        
        print("=" * 40)
        print(f"📊 Security Test Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests >= total_tests * 0.7:  # 70% pass rate
            print("🎉 Security posture looks good!")
        else:
            print("⚠️  Security improvements needed")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "results": self.test_results
        }


def main():
    """Run security tests"""
    tester = APISecurityTester()
    results = tester.run_all_tests()
    return results


if __name__ == "__main__":
    main()