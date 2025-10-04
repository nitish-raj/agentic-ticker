"""
Comprehensive security validation framework for Agentic Ticker.

This module provides enterprise-grade input validation, sanitization, and
security controls to prevent injection attacks, data corruption, and other
security vulnerabilities.
"""

import re
import html
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import unicodedata

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Security validation levels."""
    PERMISSIVE = "permissive"  # Basic validation, allows most valid inputs
    STANDARD = "standard"      # Standard security validation
    STRICT = "strict"          # Maximum security, restrictive validation


class InputType(Enum):
    """Types of input data for validation."""
    TICKER_SYMBOL = "ticker_symbol"
    COMPANY_NAME = "company_name"
    SEARCH_QUERY = "search_query"
    NUMERIC_VALUE = "numeric_value"
    DATE_STRING = "date_string"
    ALPHANUMERIC = "alphanumeric"
    TEXT_INPUT = "text_input"


@dataclass
class ValidationRule:
    """Validation rule configuration."""
    input_type: InputType
    min_length: int = 0
    max_length: int = 1000
    allowed_chars: Optional[str] = None
    forbidden_chars: Optional[str] = None
    forbidden_patterns: Optional[List[str]] = None
    required_patterns: Optional[List[str]] = None
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    custom_validator: Optional[Callable[[str], bool]] = None


class SecurityValidator:
    """
    Comprehensive security validation and sanitization system.
    
    Features:
    - Whitelist-based validation
    - Injection attack prevention
    - Length and character restrictions
    - Pattern-based validation
    - Input sanitization and normalization
    """
    
    # Predefined validation rules for different input types
    VALIDATION_RULES = {
        InputType.TICKER_SYMBOL: ValidationRule(
            input_type=InputType.TICKER_SYMBOL,
            min_length=1,
            max_length=10,
            allowed_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            required_patterns=[r'^[A-Z0-9]{1,5}(-[A-Z0-9]{1,3})?$'],
            validation_level=ValidationLevel.STRICT
        ),
        
        InputType.COMPANY_NAME: ValidationRule(
            input_type=InputType.COMPANY_NAME,
            min_length=1,
            max_length=100,
            allowed_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .&'-",
            forbidden_patterns=[
                r'<script.*?>.*?</script>',
                r'javascript:',
                r'vbscript:',
                r'onload=',
                r'onerror=',
                r'onclick=',
                r'onmouseover=',
                r'data:text/html',
                r'eval\s*\(',
                r'exec\s*\(',
                r'system\s*\(',
                r'drop\s+table',
                r'union\s+select'
            ],
            validation_level=ValidationLevel.STANDARD
        ),
        
        InputType.SEARCH_QUERY: ValidationRule(
            input_type=InputType.SEARCH_QUERY,
            min_length=1,
            max_length=200,
            forbidden_chars="<>\"'&;\\",
            forbidden_patterns=[
                r'<script.*?>.*?</script>',
                r'javascript:',
                r'vbscript:',
                r'data:text/html',
                r'eval\s*\(',
                r'exec\s*\(',
                r'system\s*\(',
                r'\$\(',
                r'`.*?`',
                r'\|\s*',
                r'&&\s*',
                r';\s*',
                r'>>\s*',
                r'<\s*'
            ],
            validation_level=ValidationLevel.STANDARD
        ),
        
        InputType.NUMERIC_VALUE: ValidationRule(
            input_type=InputType.NUMERIC_VALUE,
            min_length=1,
            max_length=20,
            allowed_chars="0123456789.-",
            required_patterns=[r'^-?\d+\.?\d*$'],
            validation_level=ValidationLevel.STRICT
        ),
        
        InputType.DATE_STRING: ValidationRule(
            input_type=InputType.DATE_STRING,
            min_length=8,
            max_length=10,
            allowed_chars="0123456789-",
            required_patterns=[r'^\d{4}-\d{2}-\d{2}$'],
            validation_level=ValidationLevel.STRICT
        ),
        
        InputType.ALPHANUMERIC: ValidationRule(
            input_type=InputType.ALPHANUMERIC,
            min_length=1,
            max_length=50,
            allowed_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            validation_level=ValidationLevel.STANDARD
        ),
        
        InputType.TEXT_INPUT: ValidationRule(
            input_type=InputType.TEXT_INPUT,
            min_length=1,
            max_length=500,
            forbidden_chars="<>\0",
            forbidden_patterns=[
                r'<script.*?>.*?</script>',
                r'javascript:',
                r'vbscript:',
                r'data:text/html',
                r'eval\s*\(',
                r'exec\s*\(',
                r'system\s*\('
            ],
            validation_level=ValidationLevel.PERMISSIVE
        )
    }
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize the security validator.
        
        Args:
            validation_level: Default validation level for operations
        """
        self.validation_level = validation_level
        self.validation_cache = {}
        
    def validate_input(
        self, 
        input_value: str, 
        input_type: InputType,
        validation_level: Optional[ValidationLevel] = None
    ) -> tuple[bool, str, str]:
        """
        Validate input according to security rules.
        
        Args:
            input_value: The input value to validate
            input_type: Type of input for validation rules
            validation_level: Override default validation level
            
        Returns:
            Tuple of (is_valid, sanitized_value, error_message)
        """
        if validation_level is None:
            validation_level = self.validation_level
            
        # Get validation rule
        rule = self.VALIDATION_RULES.get(input_type)
        if not rule:
            return False, "", f"No validation rule found for input type: {input_type}"
        
        # Check if rule level meets requirements
        level_hierarchy = {
            ValidationLevel.PERMISSIVE: 0,
            ValidationLevel.STANDARD: 1,
            ValidationLevel.STRICT: 2
        }
        
        if level_hierarchy[rule.validation_level] < level_hierarchy[validation_level]:
            # Use stricter validation
            rule = ValidationRule(
                input_type=rule.input_type,
                min_length=rule.min_length,
                max_length=max(rule.max_length // 2, 10),
                allowed_chars=rule.allowed_chars,
                forbidden_chars=rule.forbidden_chars,
                forbidden_patterns=rule.forbidden_patterns,
                required_patterns=rule.required_patterns,
                validation_level=validation_level,
                custom_validator=rule.custom_validator
            )
        
        try:
            # Convert to string and normalize
            if input_value is None:
                return False, "", "Input value cannot be None"
                
            input_str = str(input_value)
            
            # Unicode normalization to prevent homograph attacks
            input_str = unicodedata.normalize('NFKC', input_str)
            
            # Length validation
            if len(input_str) < rule.min_length:
                return False, "", f"Input too short. Minimum length: {rule.min_length}"
                
            if len(input_str) > rule.max_length:
                return False, "", f"Input too long. Maximum length: {rule.max_length}"
            
            # Character validation
            if rule.allowed_chars:
                if not all(char in rule.allowed_chars for char in input_str):
                    return False, "", "Input contains invalid characters"
            
            if rule.forbidden_chars:
                if any(char in rule.forbidden_chars for char in input_str):
                    return False, "", "Input contains forbidden characters"
            
            # Pattern validation
            if rule.forbidden_patterns:
                for pattern in rule.forbidden_patterns:
                    if re.search(pattern, input_str, re.IGNORECASE | re.MULTILINE | re.DOTALL):
                        return False, "", "Input contains forbidden pattern"
            
            if rule.required_patterns:
                for pattern in rule.required_patterns:
                    if not re.match(pattern, input_str, re.IGNORECASE):
                        return False, "", f"Input does not match required pattern: {pattern}"
            
            # Custom validation
            if rule.custom_validator:
                if not rule.custom_validator(input_str):
                    return False, "", "Input failed custom validation"
            
            # Sanitize input
            sanitized = self._sanitize_input(input_str, input_type)
            
            return True, sanitized, ""
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, "", "Validation failed due to internal error"
    
    def _sanitize_input(self, input_value: str, input_type: InputType) -> str:
        """
        Sanitize input value based on type.
        
        Args:
            input_value: Input value to sanitize
            input_type: Type of input for sanitization rules
            
        Returns:
            Sanitized input value
        """
        if input_type in [InputType.TICKER_SYMBOL, InputType.ALPHANUMERIC]:
            # For tickers and alphanumeric, just normalize case
            return input_value.strip().upper()
        
        elif input_type == InputType.COMPANY_NAME:
            # For company names, preserve case but clean extra spaces
            return ' '.join(input_value.split())
        
        elif input_type == InputType.SEARCH_QUERY:
            # For search queries, escape HTML and normalize
            sanitized = html.escape(input_value)
            return sanitized.strip()
        
        elif input_type == InputType.TEXT_INPUT:
            # For general text, escape HTML and normalize
            sanitized = html.escape(input_value)
            return sanitized.strip()
        
        else:
            # Default sanitization
            return input_value.strip()
    
    def validate_ticker_symbol(self, ticker: str) -> tuple[bool, str, str]:
        """Validate ticker symbol with strict security rules."""
        return self.validate_input(ticker, InputType.TICKER_SYMBOL)
    
    def validate_company_name(self, name: str) -> tuple[bool, str, str]:
        """Validate company name with security rules."""
        return self.validate_input(name, InputType.COMPANY_NAME)
    
    def validate_search_query(self, query: str) -> tuple[bool, str, str]:
        """Validate search query with injection prevention."""
        return self.validate_input(query, InputType.SEARCH_QUERY)
    
    def validate_numeric_value(self, value: Union[str, int, float]) -> tuple[bool, str, str]:
        """Validate numeric input."""
        return self.validate_input(str(value), InputType.NUMERIC_VALUE)
    
    def sanitize_for_prompt(self, text: str) -> str:
        """
        Sanitize text for use in AI prompts to prevent prompt injection.
        
        Args:
            text: Text to sanitize for prompt use
            
        Returns:
            Sanitized text safe for prompt injection
        """
        if not text:
            return ""
        
        # Remove potential prompt injection patterns
        sanitized = text
        
        # Remove instruction-like patterns
        injection_patterns = [
            r'(?i)ignore\s+previous\s+instructions',
            r'(?i)disregard\s+above',
            r'(?i)new\s+instruction',
            r'(?i)system\s*:',
            r'(?i)assistant\s*:',
            r'(?i)user\s*:',
            r'(?i)role\s*:',
            r'(?i)act\s+as',
            r'(?i)pretend\s+to\s+be',
            r'(?i)you\s+are\s+now',
            r'(?i)from\s+now\s+on',
            r'(?i)\{\{.*?\}\}',
            r'(?i)\[\[.*?\]\]',
            r'(?i)<<.*?>>',
            r'(?i)`.*?`',
            r'(?i)\$\{.*?\}',
        ]
        
        for pattern in injection_patterns:
            sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)
        
        # Limit length for prompt safety
        max_prompt_length = 1000
        if len(sanitized) > max_prompt_length:
            sanitized = sanitized[:max_prompt_length] + "..."
        
        # Escape special characters
        sanitized = html.escape(sanitized)
        
        return sanitized
    
    def validate_api_response(self, response_data: Any) -> tuple[bool, Any]:
        """
        Validate API response data for security.
        
        Args:
            response_data: Response data to validate
            
        Returns:
            Tuple of (is_valid, sanitized_data)
        """
        try:
            if isinstance(response_data, str):
                # Validate string responses
                is_valid, sanitized, error = self.validate_input(
                    response_data, InputType.TEXT_INPUT
                )
                return is_valid, sanitized
            
            elif isinstance(response_data, dict):
                # Recursively validate dictionary
                sanitized_dict = {}
                for key, value in response_data.items():
                    is_valid, sanitized_value = self.validate_api_response(value)
                    if not is_valid:
                        return False, None
                    sanitized_dict[key] = sanitized_value
                return True, sanitized_dict
            
            elif isinstance(response_data, list):
                # Recursively validate list
                sanitized_list = []
                for item in response_data:
                    is_valid, sanitized_item = self.validate_api_response(item)
                    if not is_valid:
                        return False, None
                    sanitized_list.append(sanitized_item)
                return True, sanitized_list
            
            else:
                # For other types, return as-is
                return True, response_data
                
        except Exception as e:
            logger.error(f"API response validation error: {str(e)}")
            return False, None


# Global validator instance
_security_validator = None


def get_security_validator(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> SecurityValidator:
    """Get or create the global security validator instance."""
    global _security_validator
    if _security_validator is None or _security_validator.validation_level != validation_level:
        _security_validator = SecurityValidator(validation_level)
    return _security_validator


# Convenience functions for common validation tasks
def validate_ticker(ticker: str) -> tuple[bool, str, str]:
    """Validate ticker symbol."""
    return get_security_validator().validate_ticker_symbol(ticker)


def validate_company_name(name: str) -> tuple[bool, str, str]:
    """Validate company name."""
    return get_security_validator().validate_company_name(name)


def validate_search_query(query: str) -> tuple[bool, str, str]:
    """Validate search query."""
    return get_security_validator().validate_search_query(query)


def sanitize_for_prompt(text: str) -> str:
    """Sanitize text for AI prompt use."""
    return get_security_validator().sanitize_for_prompt(text)


def validate_numeric_input(value: Union[str, int, float]) -> tuple[bool, str, str]:
    """Validate numeric input."""
    return get_security_validator().validate_numeric_value(value)


def validate_api_response(response_data: Any) -> tuple[bool, Any]:
    """Validate API response data."""
    return get_security_validator().validate_api_response(response_data)


# Decorator for automatic input validation
def secure_validate(
    input_type: InputType,
    param_name: str,
    validation_level: ValidationLevel = ValidationLevel.STANDARD
):
    """
    Decorator for automatic input validation of function parameters.
    
    Args:
        input_type: Type of input to validate
        param_name: Name of parameter to validate
        validation_level: Validation level to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            if param_name in bound_args.arguments:
                value = bound_args.arguments[param_name]
                validator = get_security_validator(validation_level)
                is_valid, sanitized, error = validator.validate_input(value, input_type)
                
                if not is_valid:
                    raise ValueError(f"Invalid {param_name}: {error}")
                
                # Replace with sanitized value
                bound_args.arguments[param_name] = sanitized
            
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator