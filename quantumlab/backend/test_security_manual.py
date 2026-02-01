#!/usr/bin/env python3
"""
Test security features of the API server.
"""

import requests
import time

BASE_URL = "http://localhost:8000"

def test_security_headers():
    """Test that security headers are present"""
    print("Testing Security Headers...")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/models")
    
    headers_to_check = [
        "X-Frame-Options",
        "X-Content-Type-Options",
        "Content-Security-Policy",
        "Referrer-Policy",
    ]
    
    for header in headers_to_check:
        if header in response.headers:
            print(f"✓ {header}: {response.headers[header][:50]}...")
        else:
            print(f"✗ {header}: Missing")
    
    print()

def test_rate_limiting():
    """Test rate limiting"""
    print("Testing Rate Limiting...")
    print("=" * 60)
    
    # Make many rapid requests
    for i in range(65):
        response = requests.get(f"{BASE_URL}/elements")
        if response.status_code == 429:
            print(f"✓ Rate limit triggered after {i} requests")
            print(f"  Response: {response.json()}")
            break
        elif i % 20 == 0:
            print(f"  Request {i+1}: OK")
    else:
        print("✗ Rate limit not triggered after 65 requests")
    
    print()

def test_input_validation():
    """Test input validation"""
    print("Testing Input Validation...")
    print("=" * 60)
    
    # Test SQL injection
    malicious_inputs = [
        {"tags": ["'; DROP TABLE users--"]},
        {"notes": "<script>alert('XSS')</script>"},
    ]
    
    for payload in malicious_inputs:
        try:
            response = requests.post(
                f"{BASE_URL}/library/save",
                json={
                    "molecule_data": {"atoms": [], "positions": []},
                    "metadata": {},
                    **payload
                }
            )
            if response.status_code == 400:
                print(f"✓ Blocked malicious input: {list(payload.values())[0]}")
            else:
                print(f"✗ Did not block: {payload}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   SECURITY FEATURES TEST SUITE")
    print("=" * 60 + "\n")
    
    test_security_headers()
    test_input_validation()
    test_rate_limiting()  # Last because it makes many requests
    
    print("=" * 60)
    print("Test suite complete!")
    print("=" * 60)
