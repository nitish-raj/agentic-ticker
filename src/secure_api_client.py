"""
Secure API client for external service integration.

This module provides a secure way to make API calls without exposing
API keys in URLs, implementing proper authentication headers,
rate limiting, and comprehensive error handling.
"""

import time
import requests
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading
from collections import defaultdict, deque

from .sanitization import sanitize_url
from .config import get_config

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class APIAuthenticationError(Exception):
    """Raised when API authentication fails."""

    pass


@dataclass
class RateLimiter:
    """Thread-safe rate limiter using sliding window algorithm."""

    max_requests: int = 100
    window_seconds: int = 60

    def __post_init__(self):
        self._requests: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, key: str = "default") -> bool:
        """Check if request is allowed based on rate limit."""
        with self._lock:
            now = time.time()
            requests = self._requests[key]

            # Remove old requests outside the window
            while requests and requests[0] <= now - self.window_seconds:
                requests.popleft()

            # Check if under limit
            if len(requests) < self.max_requests:
                requests.append(now)
                return True

            return False

    def get_wait_time(self, key: str = "default") -> float:
        """Get time to wait before next request is allowed."""
        with self._lock:
            requests = self._requests[key]
            if not requests:
                return 0.0

            oldest_request = requests[0]
            wait_time = self.window_seconds - (time.time() - oldest_request)
            return max(0.0, wait_time)


@dataclass
class APIEndpoint:
    """Configuration for an API endpoint."""

    name: str
    base_url: str
    api_key_header: str = "X-API-Key"
    api_key_env_var: str = ""
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    def get_api_key(self) -> str:
        """Get API key from config (config.yaml only - no environment variables)."""
        # Try config first (repository policy: config.yaml only)
        try:
            config = get_config()
            if hasattr(config, "gemini") and self.name.lower() == "gemini":
                return config.gemini.api_key
            elif hasattr(config, "coingecko") and self.name.lower() == "coingecko":
                return config.coingecko.demo_api_key or config.coingecko.pro_api_key
        except Exception:
            pass

        return ""


class SecureAPIClient:
    """
    Secure API client with proper authentication, rate limiting,
    and comprehensive error handling.
    """

    def __init__(self):
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.session = requests.Session()

        # Configure session with security defaults
        self.session.headers.update(
            {
                "User-Agent": "Agentic-Ticker/1.0 (Secure Client)",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )

        # Initialize endpoints
        self.endpoints = {
            "gemini": APIEndpoint(
                name="gemini",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                api_key_header="x-goog-api-key",
                api_key_env_var="",  # No environment variables - config.yaml only
                timeout=30,
                max_retries=3,
                rate_limit_requests=60,
                rate_limit_window=60,
            ),
            "coingecko": APIEndpoint(
                name="coingecko",
                base_url="https://api.coingecko.com/api/v3",
                api_key_header="x-cg-demo-api-key",
                api_key_env_var="",  # No environment variables - config.yaml only
                timeout=30,
                max_retries=3,
                rate_limit_requests=30,
                rate_limit_window=60,
            ),
            "yahoo_finance": APIEndpoint(
                name="yahoo_finance",
                base_url="https://query1.finance.yahoo.com",
                api_key_header="",  # Yahoo Finance doesn't use API keys
                timeout=30,
                max_retries=3,
                rate_limit_requests=100,
                rate_limit_window=60,
            ),
            "duckduckgo": APIEndpoint(
                name="duckduckgo",
                base_url="https://duckduckgo.com",
                api_key_header="",  # DDGS doesn't use API keys
                timeout=30,
                max_retries=2,
                rate_limit_requests=30,
                rate_limit_window=60,
            ),
        }

        # Initialize rate limiters
        for name, endpoint in self.endpoints.items():
            self.rate_limiters[name] = RateLimiter(
                max_requests=endpoint.rate_limit_requests,
                window_seconds=endpoint.rate_limit_window,
            )

    def _get_rate_limiter(self, endpoint_name: str) -> RateLimiter:
        """Get rate limiter for endpoint."""
        if endpoint_name in self.rate_limiters:
            return self.rate_limiters[endpoint_name]
        else:
            # Return a default rate limiter for unknown endpoints
            return RateLimiter(max_requests=100, window_seconds=60)

    def _prepare_headers(
        self, endpoint: APIEndpoint, additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Prepare secure headers for API request."""
        headers = {}

        # Add API key if required
        if endpoint.api_key_header:
            api_key = endpoint.get_api_key()
            if api_key:
                headers[endpoint.api_key_header] = api_key
            else:
                logger.warning(f"No API key available for {endpoint.name}")

        # Add additional headers
        if additional_headers:
            headers.update(additional_headers)

        # Add security headers
        headers.update(
            {
                "X-Requested-With": "Agentic-Ticker",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )

        return headers

    def _sanitize_url_params(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Sanitize URL and parameters to prevent API key exposure."""
        # Remove any API keys from URL parameters
        if params:
            sanitized_params = {}
            for key, value in params.items():
                if (
                    "key" in key.lower()
                    or "secret" in key.lower()
                    or "token" in key.lower()
                ):
                    sanitized_params[key] = "[REDACTED]"
                else:
                    sanitized_params[key] = value
            params = sanitized_params

        # Sanitize URL itself
        url = sanitize_url(url)

        return url, params

    def _make_request(
        self,
        method: str,
        endpoint_name: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> requests.Response:
        """
        Make a secure API request with proper authentication and rate limiting.
        """
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")

        # Check rate limiting
        rate_limiter = self._get_rate_limiter(endpoint_name)
        if not rate_limiter.is_allowed():
            wait_time = rate_limiter.get_wait_time()
            raise RateLimitError(
                f"Rate limit exceeded for {endpoint_name}. Wait {wait_time:.1f} seconds."
            )

        # Prepare URL and headers
        url = f"{endpoint.base_url.rstrip('/')}/{path.lstrip('/')}"
        url, params = self._sanitize_url_params(url, params)

        request_headers = self._prepare_headers(endpoint, headers)

        # Set timeout
        request_timeout = timeout or endpoint.timeout
        request_max_retries = max_retries or endpoint.max_retries

        # Make request with retries
        last_exception = None
        for attempt in range(request_max_retries + 1):
            try:
                response = self.session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=request_headers,
                    timeout=request_timeout,
                )

                # Log successful request (without sensitive data)
                logger.debug(f"{method.upper()} {url} -> {response.status_code}")

                # Handle authentication errors
                if response.status_code == 401:
                    raise APIAuthenticationError(
                        f"Authentication failed for {endpoint_name}"
                    )
                elif response.status_code == 403:
                    raise APIAuthenticationError(
                        f"Access forbidden for {endpoint_name}"
                    )

                # Handle rate limiting from server
                if response.status_code == 429:
                    retry_after = response.headers.get(
                        "Retry-After", endpoint.retry_delay
                    )
                    wait_time = float(retry_after)
                    if attempt < request_max_retries:
                        logger.warning(
                            f"Rate limited by {endpoint_name}, waiting {wait_time}s"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RateLimitError(
                            f"Rate limit exceeded for {endpoint_name} after {request_max_retries} retries"
                        )

                # Raise for other HTTP errors
                response.raise_for_status()
                return response

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < request_max_retries:
                    logger.warning(
                        f"Timeout for {endpoint_name}, retrying ({attempt + 1}/{request_max_retries})"
                    )
                    time.sleep(
                        endpoint.retry_delay * (2**attempt)
                    )  # Exponential backoff
                    continue
                else:
                    logger.error(
                        f"Timeout for {endpoint_name} after {request_max_retries} retries"
                    )
                    raise

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < request_max_retries:
                    logger.warning(
                        f"Connection error for {endpoint_name}, retrying ({attempt + 1}/{request_max_retries})"
                    )
                    time.sleep(endpoint.retry_delay * (2**attempt))
                    continue
                else:
                    logger.error(
                        f"Connection error for {endpoint_name} after {request_max_retries} retries"
                    )
                    raise

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < request_max_retries:
                    logger.warning(
                        f"Request error for {endpoint_name}, retrying ({attempt + 1}/{request_max_retries})"
                    )
                    time.sleep(endpoint.retry_delay)
                    continue
                else:
                    logger.error(
                        f"Request error for {endpoint_name} after {request_max_retries} retries"
                    )
                    raise

        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise requests.exceptions.RequestException(
                f"Request failed for {endpoint_name}"
            )

    def get(self, endpoint_name: str, path: str, **kwargs) -> requests.Response:
        """Make a GET request."""
        return self._make_request("GET", endpoint_name, path, **kwargs)

    def post(self, endpoint_name: str, path: str, **kwargs) -> requests.Response:
        """Make a POST request."""
        return self._make_request("POST", endpoint_name, path, **kwargs)

    def put(self, endpoint_name: str, path: str, **kwargs) -> requests.Response:
        """Make a PUT request."""
        return self._make_request("PUT", endpoint_name, path, **kwargs)

    def delete(self, endpoint_name: str, path: str, **kwargs) -> requests.Response:
        """Make a DELETE request."""
        return self._make_request("DELETE", endpoint_name, path, **kwargs)

    def close(self):
        """Close the session."""
        self.session.close()


# Global secure API client instance
_secure_client: Optional[SecureAPIClient] = None
_client_lock = threading.Lock()


def get_secure_client() -> SecureAPIClient:
    """Get the global secure API client instance."""
    global _secure_client
    if _secure_client is None:
        with _client_lock:
            if _secure_client is None:
                _secure_client = SecureAPIClient()
    return _secure_client


def close_secure_client():
    """Close the global secure API client."""
    global _secure_client
    if _secure_client is not None:
        _secure_client.close()
        _secure_client = None


# Convenience functions for specific APIs
def secure_gemini_request(
    path: str, method: str = "POST", **kwargs
) -> requests.Response:
    """Make a secure request to Gemini API."""
    client = get_secure_client()
    if method.upper() == "GET":
        return client.get("gemini", path, **kwargs)
    elif method.upper() == "POST":
        return client.post("gemini", path, **kwargs)
    else:
        return client._make_request(method, "gemini", path, **kwargs)


def secure_coingecko_request(
    path: str, method: str = "GET", **kwargs
) -> requests.Response:
    """Make a secure request to CoinGecko API."""
    client = get_secure_client()
    if method.upper() == "GET":
        return client.get("coingecko", path, **kwargs)
    elif method.upper() == "POST":
        return client.post("coingecko", path, **kwargs)
    else:
        return client._make_request(method, "coingecko", path, **kwargs)


def secure_yahoo_request(path: str, method: str = "GET", **kwargs) -> requests.Response:
    """Make a secure request to Yahoo Finance API."""
    client = get_secure_client()
    if method.upper() == "GET":
        return client.get("yahoo_finance", path, **kwargs)
    elif method.upper() == "POST":
        return client.post("yahoo_finance", path, **kwargs)
    else:
        return client._make_request(method, "yahoo_finance", path, **kwargs)
