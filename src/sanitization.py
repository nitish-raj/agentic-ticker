"""
Centralized sanitization utilities for Agentic Ticker.

This module provides comprehensive sanitization functions to prevent API key exposure
in error messages, logs, debug output, and any other text output.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Sanitizer:
    """Centralized sanitization utility for sensitive information."""
    
    # Patterns for detecting sensitive information
    API_KEY_PATTERNS = [
        # Gemini API keys in URLs (most specific first)
        r'key=[a-zA-Z0-9_\-]{10,}',
        # Environment variable assignments (most specific first)
        r'(GEMINI_API_KEY|COINGECKO_DEMO_API_KEY|COINGECKO_API_KEY)\s*=\s*["\']?[^"\'\s]+["\']?',
        # API keys in headers (common patterns)
        r'(?:api[_-]?key|apikey|api[_-]?secret)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{16,})["\']?',
        # Bearer tokens
        r'Bearer\s+([a-zA-Z0-9_\-\.]{16,})',
        # Generic API key patterns (hex, base64-like) - less aggressive
        r'["\']([a-zA-Z0-9+/]{32,}={0,2})["\']',
    ]
    
    # Replacement text for sensitive information
    REPLACEMENTS = {
        'api_key': '[REDACTED_API_KEY]',
        'token': '[REDACTED_TOKEN]',
        'secret': '[REDACTED_SECRET]',
        'bearer': '[REDACTED_BEARER_TOKEN]',
        'env_var': '[REDACTED_CREDENTIAL]',
    }
    
    @classmethod
    def sanitize_api_keys(cls, text: str) -> str:
        """
        Remove API keys and other sensitive information from text.
        
        Args:
            text: Input text that may contain sensitive information
            
        Returns:
            Sanitized text with API keys replaced
        """
        if not text or not isinstance(text, str):
            return text
            
        sanitized = text
        
        # Apply patterns in order - most specific first
        # Environment variable assignments
        sanitized = re.sub(
            r'(GEMINI_API_KEY|COINGECKO_DEMO_API_KEY|COINGECKO_API_KEY)\s*=\s*["\']?[^"\'\s]+["\']?',
            lambda m: f"{m.group(1)}=[REDACTED_CREDENTIAL]",
            sanitized,
            flags=re.IGNORECASE
        )
        
        # API keys in URLs (more permissive to catch within text)
        sanitized = re.sub(r'key=[a-zA-Z0-9_\-\.]{1,}', 'key=[REDACTED]', sanitized, flags=re.IGNORECASE | re.MULTILINE)
        
        # Bearer tokens
        sanitized = re.sub(r'Bearer\s+[a-zA-Z0-9_\-\.]{8,}', 'Bearer [REDACTED_TOKEN]', sanitized, flags=re.IGNORECASE)
        
        # API keys in headers
        sanitized = re.sub(
            r'(?:api[_-]?key|apikey|api[_-]?secret)\s*[:=]\s*["\']?[a-zA-Z0-9_\-]{16,}["\']?',
            lambda m: f"{m.group(0).split(':')[0].split('=')[0]}=[REDACTED]" if ':' in m.group(0) or '=' in m.group(0) else '[REDACTED]',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Generic long alphanumeric strings that might be keys
        sanitized = re.sub(r'["\'][a-zA-Z0-9+/]{32,}={0,2}["\']', '[REDACTED]', sanitized)
                
        return sanitized
    

    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
        """
        Sanitize dictionary values that may contain sensitive information.
        
        Args:
            data: Dictionary to sanitize
            deep: Whether to recursively sanitize nested dictionaries
            
        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            return data
            
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = cls.sanitize_api_keys(value)
            elif isinstance(value, dict) and deep:
                sanitized[key] = cls.sanitize_dict(value, deep=True)
            elif isinstance(value, list) and deep:
                sanitized[key] = cls.sanitize_list(value, deep=True)
            else:
                sanitized[key] = value
                
        return sanitized
    
    @classmethod
    def sanitize_list(cls, data: List[Any], deep: bool = True) -> List[Any]:
        """
        Sanitize list values that may contain sensitive information.
        
        Args:
            data: List to sanitize
            deep: Whether to recursively sanitize nested structures
            
        Returns:
            Sanitized list
        """
        if not isinstance(data, list):
            return data
            
        sanitized = []
        for item in data:
            if isinstance(item, str):
                sanitized.append(cls.sanitize_api_keys(item))
            elif isinstance(item, dict) and deep:
                sanitized.append(cls.sanitize_dict(item, deep=True))
            elif isinstance(item, list) and deep:
                sanitized.append(cls.sanitize_list(item, deep=True))
            else:
                sanitized.append(item)
                
        return sanitized
    
    @classmethod
    def sanitize_error_message(cls, error_msg: Union[str, Exception]) -> str:
        """
        Sanitize error messages to prevent API key exposure.
        
        Args:
            error_msg: Error message or exception
            
        Returns:
            Sanitized error message
        """
        if isinstance(error_msg, Exception):
            error_msg = str(error_msg)
            
        return cls.sanitize_api_keys(str(error_msg))
    
    @classmethod
    def sanitize_url(cls, url: str) -> str:
        """
        Sanitize URLs that may contain API keys.
        
        Args:
            url: URL that may contain API keys
            
        Returns:
            Sanitized URL
        """
        if not url or not isinstance(url, str):
            return url
            
        # Specific URL sanitization
        sanitized = re.sub(r'key=[^&\s]+', 'key=[REDACTED]', url, flags=re.IGNORECASE)
        sanitized = re.sub(r'token=[^&\s]+', 'token=[REDACTED]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'secret=[^&\s]+', 'secret=[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @classmethod
    def sanitize_log_message(cls, message: str, level: str = "INFO") -> str:
        """
        Sanitize log messages to prevent sensitive information exposure.
        
        Args:
            message: Log message to sanitize
            level: Log level (for additional filtering if needed)
            
        Returns:
            Sanitized log message
        """
        return cls.sanitize_api_keys(message)
    
    @classmethod
    def sanitize_debug_output(cls, output: Any) -> Any:
        """
        Sanitize debug output that may contain sensitive information.
        
        Args:
            output: Debug output to sanitize
            
        Returns:
            Sanitized debug output
        """
        if isinstance(output, str):
            return cls.sanitize_api_keys(output)
        elif isinstance(output, dict):
            return cls.sanitize_dict(output)
        elif isinstance(output, list):
            return cls.sanitize_list(output)
        else:
            return output


# Convenience functions for backward compatibility and ease of use
def sanitize_error_message(error_msg: Union[str, Exception]) -> str:
    """Convenience function to sanitize error messages."""
    return Sanitizer.sanitize_error_message(error_msg)


def sanitize_api_keys(text: str) -> str:
    """Convenience function to sanitize API keys in text."""
    return Sanitizer.sanitize_api_keys(text)


def sanitize_url(url: str) -> str:
    """Convenience function to sanitize URLs."""
    return Sanitizer.sanitize_url(url)


def sanitize_log_message(message: str, level: str = "INFO") -> str:
    """Convenience function to sanitize log messages."""
    return Sanitizer.sanitize_log_message(message, level)


def sanitize_for_debug(output: Any) -> Any:
    """Convenience function to sanitize debug output."""
    return Sanitizer.sanitize_debug_output(output)


# Decorator for automatic sanitization of function return values
def sanitize_output(func):
    """
    Decorator to automatically sanitize function output.
    
    This decorator will sanitize any string output from functions to prevent
    accidental API key exposure in return values.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Sanitize based on return type
        if isinstance(result, str):
            return Sanitizer.sanitize_api_keys(result)
        elif isinstance(result, dict):
            return Sanitizer.sanitize_dict(result)
        elif isinstance(result, list):
            return Sanitizer.sanitize_list(result)
        else:
            return result
            
    return wrapper


# Decorator for sanitizing exceptions
def sanitize_exceptions(func):
    """
    Decorator to automatically sanitize exception messages.
    
    This decorator will catch exceptions and sanitize their messages before
    re-raising them to prevent API key exposure in error messages.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Create a new exception with sanitized message
            sanitized_msg = Sanitizer.sanitize_error_message(str(e))
            
            # Preserve the original exception type
            new_exception = type(e)(sanitized_msg)
            raise new_exception from e
            
    return wrapper