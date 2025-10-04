"""
Security middleware for Agentic Ticker application.

This module provides comprehensive security middleware including:
- Security headers enforcement
- Rate limiting
- Request size limits
- CORS security enhancements
- IP-based filtering
- Request validation
"""

import time
import hashlib
import ipaddress
from typing import Dict, List, Optional, Set, Callable, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse
import logging
import re
import os

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    def __init__(self, app, https_enabled: bool = True):
        super().__init__(app)
        self.https_enabled = https_enabled
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Security headers
        headers = response.headers
        
        # HSTS (HTTP Strict Transport Security) - only if HTTPS is enabled
        if self.https_enabled:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Content Security Policy
        headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self' https://api.coingecko.com https://generativelanguage.googleapis.com; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        # Other security headers
        headers["X-Frame-Options"] = "DENY"
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-XSS-Protection"] = "1; mode=block"
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )
        
        # Remove server information
        headers.pop("Server", None)
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with multiple strategies."""
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
        whitelist_ips: Optional[List[str]] = None,
        blacklist_ips: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.whitelist_ips = set(whitelist_ips or [])
        self.blacklist_ips = set(blacklist_ips or [])
        
        # Rate limiting storage
        self.minute_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.hour_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.burst_requests: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Cleanup task
        self.last_cleanup = time.time()
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxies."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        return request.client.host if request.client else "unknown"
    
    def _is_ip_allowed(self, ip: str) -> bool:
        """Check if IP is allowed based on whitelist/blacklist."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check blacklist first
            for blacklisted in self.blacklist_ips:
                try:
                    if ip_obj in ipaddress.ip_network(blacklisted, strict=False):
                        return False
                except ValueError:
                    # Not a network, treat as single IP
                    if ip == blacklisted:
                        return False
            
            # Check whitelist
            if self.whitelist_ips:
                for whitelisted in self.whitelist_ips:
                    try:
                        if ip_obj in ipaddress.ip_network(whitelisted, strict=False):
                            return True
                    except ValueError:
                        # Not a network, treat as single IP
                        if ip == whitelisted:
                            return True
                return False  # Not in whitelist
            
            return True
        except ValueError:
            # Invalid IP, block it
            return False
    
    def _cleanup_old_requests(self):
        """Clean up old request records."""
        now = time.time()
        
        # Clean up every minute
        if now - self.last_cleanup < 60:
            return
        
        self.last_cleanup = now
        
        # Clean minute window
        cutoff_minute = now - 60
        for ip, requests in list(self.minute_requests.items()):
            while requests and requests[0] < cutoff_minute:
                requests.popleft()
            if not requests:
                del self.minute_requests[ip]
        
        # Clean hour window
        cutoff_hour = now - 3600
        for ip, requests in list(self.hour_requests.items()):
            while requests and requests[0] < cutoff_hour:
                requests.popleft()
            if not requests:
                del self.hour_requests[ip]
        
        # Clean burst window (10 seconds)
        cutoff_burst = now - 10
        for ip, requests in list(self.burst_requests.items()):
            while requests and requests[0] < cutoff_burst:
                requests.popleft()
            if not requests:
                del self.burst_requests[ip]
    
    def _check_rate_limit(self, ip: str) -> tuple[bool, Optional[str]]:
        """Check if IP is rate limited."""
        now = time.time()
        
        # Clean up old records
        self._cleanup_old_requests()
        
        # Check burst limit (10 second window)
        burst_requests = self.burst_requests[ip]
        while burst_requests and burst_requests[0] < now - 10:
            burst_requests.popleft()
        
        if len(burst_requests) >= self.burst_size:
            return False, "Burst limit exceeded"
        
        # Check minute limit
        minute_requests = self.minute_requests[ip]
        while minute_requests and minute_requests[0] < now - 60:
            minute_requests.popleft()
        
        if len(minute_requests) >= self.requests_per_minute:
            return False, "Minute rate limit exceeded"
        
        # Check hour limit
        hour_requests = self.hour_requests[ip]
        while hour_requests and hour_requests[0] < now - 3600:
            hour_requests.popleft()
        
        if len(hour_requests) >= self.requests_per_hour:
            return False, "Hour rate limit exceeded"
        
        # Record this request
        burst_requests.append(now)
        minute_requests.append(now)
        hour_requests.append(now)
        
        return True, None
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = self._get_client_ip(request)
        
        # Check IP whitelist/blacklist
        if not self._is_ip_allowed(client_ip):
            logger.warning(f"Blocked request from blacklisted IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied"}
            )
        
        # Check rate limits
        allowed, reason = self._check_rate_limit(client_ip)
        if not allowed:
            logger.warning(f"Rate limit exceeded for IP {client_ip}: {reason}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded: {reason}",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Minute"] = str(
            max(0, self.requests_per_minute - len(self.minute_requests[client_ip]))
        )
        response.headers["X-RateLimit-Remaining-Hour"] = str(
            max(0, self.requests_per_hour - len(self.hour_requests[client_ip]))
        )
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request size."""
    
    def __init__(self, app, max_size_mb: int = 10):
        super().__init__(app)
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size_bytes:
                    logger.warning(f"Request too large: {size} bytes")
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": f"Request too large. Maximum size is {self.max_size_bytes // (1024*1024)}MB"}
                    )
            except ValueError:
                pass
        
        # For streaming requests, we need to check the actual content
        # This is a simplified check - in production, you might want more sophisticated handling
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Try to get the body (this will consume it, so we need to handle it carefully)
                body = await request.body()
                if len(body) > self.max_size_bytes:
                    logger.warning(f"Request body too large: {len(body)} bytes")
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": f"Request too large. Maximum size is {self.max_size_bytes // (1024*1024)}MB"}
                    )
                
                # Create a new request with the body
                # This is a simplified approach - in production, you'd want to handle this more elegantly
                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                
                request = Request(request.scope, receive)
            except Exception as e:
                logger.error(f"Error checking request size: {e}")
        
        return await call_next(request)


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for basic input validation and sanitization."""
    
    def __init__(self, app):
        super().__init__(app)
        # Common SQL injection patterns
        self.sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"][\w\s]+['\"]\s*=\s*['\"][\w\s]+['\"])",
            r"(--|#|\/\*|\*\/)",
            r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT|ONLOAD|ONERROR)\b)",
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",
            r"javascript\s*:",
            r"on\w+\s*=",
            r"<\s*iframe[^>]*>",
            r"<\s*object[^>]*>",
            r"<\s*embed[^>]*>",
        ]
    
    def _validate_input(self, data: str) -> tuple[bool, Optional[str]]:
        """Validate input for common attacks."""
        if not data:
            return True, None
        
        # Check for SQL injection
        for pattern in self.sql_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                return False, "Potential SQL injection detected"
        
        # Check for XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                return False, "Potential XSS detected"
        
        return True, None
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Validate query parameters
        for key, value in request.query_params.items():
            is_valid, reason = self._validate_input(value)
            if not valid:
                logger.warning(f"Invalid query parameter {key}: {reason}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": f"Invalid input in parameter '{key}': {reason}"}
                )
        
        # Validate path parameters
        for key, value in request.path_params.items():
            is_valid, reason = self._validate_input(str(value))
            if not valid:
                logger.warning(f"Invalid path parameter {key}: {reason}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": f"Invalid input in path parameter '{key}': {reason}"}
                )
        
        # For POST/PUT requests, we'd need to validate the body
        # This is a simplified implementation
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    body = await request.json()
                    # Recursively validate all string values
                    def validate_dict(d):
                        for k, v in d.items():
                            if isinstance(v, str):
                                is_valid, reason = self._validate_input(v)
                                if not valid:
                                    return False, f"Invalid input in field '{k}': {reason}"
                            elif isinstance(v, dict):
                                result = validate_dict(v)
                                if result:
                                    return result
                            elif isinstance(v, list):
                                for i, item in enumerate(v):
                                    if isinstance(item, str):
                                        is_valid, reason = self._validate_input(item)
                                        if not valid:
                                            return False, f"Invalid input in field '{k}[{i}]': {reason}"
                        return None
                    
                    result = validate_dict(body)
                    if result:
                        is_valid, reason = result
                        logger.warning(f"Invalid request body: {reason}")
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={"detail": reason}
                        )
            except Exception:
                # If we can't parse the body, let the endpoint handle it
                pass
        
        return await call_next(request)


class SecureCORSMiddleware(CORSMiddleware):
    """Enhanced CORS middleware with additional security features."""
    
    def __init__(
        self,
        app,
        allow_origins: List[str] = None,
        allow_methods: List[str] = None,
        allow_headers: List[str] = None,
        allow_credentials: bool = False,
        allow_origin_regex: str = None,
        expose_headers: List[str] = None,
        max_age: int = 600,
        vary_headers: List[str] = ["Origin"],
        strict_origins: bool = True
    ):
        # Filter and validate origins
        if strict_origins and allow_origins:
            filtered_origins = []
            for origin in allow_origins:
                # Only allow HTTPS origins in production (except localhost)
                if (origin.startswith("https://") or 
                    origin.startswith(("http://localhost:", "http://127.0.0.1:"))):
                    filtered_origins.append(origin)
                else:
                    logger.warning(f"Rejecting insecure origin: {origin}")
            allow_origins = filtered_origins
        
        super().__init__(
            app,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            allow_credentials=allow_credentials,
            allow_origin_regex=allow_origin_regex,
            expose_headers=expose_headers,
            max_age=max_age,
            vary_headers=vary_headers
        )
        
        self.strict_origins = strict_origins
    
    def is_allowed_origin(self, origin: str) -> bool:
        """Enhanced origin validation."""
        if not origin:
            return False
        
        # Check exact matches first
        if origin in self.allow_origins:
            return True
        
        # Check regex patterns
        if self.allow_origin_regex and re.match(self.allow_origin_regex, origin):
            return True
        
        # Additional security checks
        if self.strict_origins:
            # Reject origins with suspicious patterns
            suspicious_patterns = [
                r".*\.tk$",
                r".*\.ml$",
                r".*\.ga$",
                r".*\.cf$",
                r".*bit\.ly.*",
                r".*tinyurl\.com.*",
            ]
            
            for pattern in suspicious_patterns:
                if re.match(pattern, origin, re.IGNORECASE):
                    logger.warning(f"Blocked suspicious origin: {origin}")
                    return False
        
        return super().is_allowed_origin(origin)


def setup_security_middleware(
    app,
    https_enabled: bool = True,
    rate_limit_enabled: bool = True,
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    max_request_size_mb: int = 10,
    whitelist_ips: Optional[List[str]] = None,
    blacklist_ips: Optional[List[str]] = None,
    cors_origins: Optional[List[str]] = None,
    strict_cors: bool = True
):
    """Set up all security middleware for the application."""
    
    # Security headers (always first)
    app.add_middleware(SecurityHeadersMiddleware, https_enabled=https_enabled)
    
    # Rate limiting
    if rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            whitelist_ips=whitelist_ips,
            blacklist_ips=blacklist_ips
        )
    
    # Request size limits
    app.add_middleware(RequestSizeLimitMiddleware, max_size_mb=max_request_size_mb)
    
    # Input validation
    app.add_middleware(InputValidationMiddleware)
    
    # CORS (last, so it applies to all responses)
    if cors_origins:
        app.add_middleware(
            SecureCORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
            allow_credentials=True,
            max_age=600,
            strict_origins=strict_cors
        )
    
    logger.info("Security middleware configured successfully")