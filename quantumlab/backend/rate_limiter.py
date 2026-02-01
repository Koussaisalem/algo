"""
Rate limiting middleware for FastAPI to prevent abuse and DDoS attacks.
"""

import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window algorithm.
    For production, use Redis-based rate limiting.
    """
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_window: Dict[str, list] = defaultdict(list)
        self.hour_window: Dict[str, list] = defaultdict(list)
        self.last_cleanup = time.time()
    
    def _cleanup_old_entries(self):
        """Remove old entries to prevent memory leaks"""
        current_time = time.time()
        
        # Cleanup every 5 minutes
        if current_time - self.last_cleanup < 300:
            return
        
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600
        
        # Cleanup minute window
        for ip in list(self.minute_window.keys()):
            self.minute_window[ip] = [t for t in self.minute_window[ip] if t > minute_cutoff]
            if not self.minute_window[ip]:
                del self.minute_window[ip]
        
        # Cleanup hour window
        for ip in list(self.hour_window.keys()):
            self.hour_window[ip] = [t for t in self.hour_window[ip] if t > hour_cutoff]
            if not self.hour_window[ip]:
                del self.hour_window[ip]
        
        self.last_cleanup = current_time
    
    def is_allowed(self, identifier: str) -> Tuple[bool, str]:
        """
        Check if request is allowed based on rate limits.
        
        Args:
            identifier: IP address or user ID
        
        Returns:
            (allowed: bool, message: str)
        """
        current_time = time.time()
        self._cleanup_old_entries()
        
        # Check minute window
        minute_cutoff = current_time - 60
        recent_minute = [t for t in self.minute_window[identifier] if t > minute_cutoff]
        
        if len(recent_minute) >= self.requests_per_minute:
            retry_after = int(60 - (current_time - recent_minute[0]))
            return False, f"Rate limit exceeded. Try again in {retry_after} seconds."
        
        # Check hour window
        hour_cutoff = current_time - 3600
        recent_hour = [t for t in self.hour_window[identifier] if t > hour_cutoff]
        
        if len(recent_hour) >= self.requests_per_hour:
            retry_after = int(3600 - (current_time - recent_hour[0]))
            return False, f"Hourly rate limit exceeded. Try again in {retry_after // 60} minutes."
        
        # Record request
        self.minute_window[identifier].append(current_time)
        self.hour_window[identifier].append(current_time)
        
        return True, "OK"
    
    def get_stats(self, identifier: str) -> Dict[str, int]:
        """Get current rate limit stats for an identifier"""
        current_time = time.time()
        
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600
        
        minute_count = len([t for t in self.minute_window[identifier] if t > minute_cutoff])
        hour_count = len([t for t in self.hour_window[identifier] if t > hour_cutoff])
        
        return {
            "requests_last_minute": minute_count,
            "requests_last_hour": hour_count,
            "minute_limit": self.requests_per_minute,
            "hour_limit": self.requests_per_hour,
            "minute_remaining": max(0, self.requests_per_minute - minute_count),
            "hour_remaining": max(0, self.requests_per_hour - hour_count),
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to apply rate limiting to all requests.
    """
    
    def __init__(self, app, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_minute, requests_per_hour)
        # Whitelist certain paths from rate limiting
        self.whitelist = ["/health", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for whitelisted paths
        if request.url.path in self.whitelist:
            return await call_next(request)
        
        # Get client identifier (IP address or user ID if authenticated)
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        allowed, message = self.limiter.is_allowed(client_ip)
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=message,
                headers={"Retry-After": "60"}
            )
        
        # Add rate limit info to response headers
        stats = self.limiter.get_stats(client_ip)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit-Minute"] = str(stats["minute_limit"])
        response.headers["X-RateLimit-Remaining-Minute"] = str(stats["minute_remaining"])
        response.headers["X-RateLimit-Limit-Hour"] = str(stats["hour_limit"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(stats["hour_remaining"])
        
        return response


# Endpoint-specific rate limiters
class StrictRateLimiter(RateLimiter):
    """Stricter rate limits for expensive operations"""
    def __init__(self):
        super().__init__(requests_per_minute=10, requests_per_hour=100)


class GenerousRateLimiter(RateLimiter):
    """More generous rate limits for read-only operations"""
    def __init__(self):
        super().__init__(requests_per_minute=120, requests_per_hour=5000)
