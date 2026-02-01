"""
Test security middleware functionality.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rate_limiter import RateLimitMiddleware, RateLimiter
from security_headers import SecurityHeadersMiddleware


def test_rate_limiter():
    """Test that rate limiter tracks requests"""
    limiter = RateLimiter(requests_per_minute=5, requests_per_hour=100)
    
    # Should allow first 5 requests
    for i in range(5):
        assert limiter.is_allowed("test_ip") == True
    
    # Should block 6th request
    assert limiter.is_allowed("test_ip") == False


def test_security_headers():
    """Test that security headers are added to responses"""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)
    
    @app.get("/test")
    def test_route():
        return {"status": "ok"}
    
    client = TestClient(app)
    response = client.get("/test")
    
    # Check security headers
    assert "X-Frame-Options" in response.headers
    assert response.headers["X-Frame-Options"] == "DENY"
    
    assert "X-Content-Type-Options" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    
    assert "Content-Security-Policy" in response.headers
    assert "Referrer-Policy" in response.headers


def test_rate_limiting_middleware():
    """Test rate limiting middleware integration"""
    app = FastAPI()
    limiter = RateLimiter(requests_per_minute=3, requests_per_hour=100)
    app.add_middleware(RateLimitMiddleware, rate_limiter=limiter)
    
    @app.get("/test")
    def test_route():
        return {"status": "ok"}
    
    client = TestClient(app)
    
    # First 3 requests should succeed
    for i in range(3):
        response = client.get("/test")
        assert response.status_code == 200
    
    # 4th request should be rate limited
    response = client.get("/test")
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
