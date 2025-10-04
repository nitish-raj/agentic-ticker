# Quickstart Guide: Code Refactoring and Modularization

## Overview
This guide provides step-by-step instructions for implementing the code refactoring and modularization feature. The goal is to eliminate code duplication and improve modularity by extracting common patterns into reusable utilities and decorators.

## Prerequisites

### System Requirements
- Python 3.11+ installed
- Git repository access
- Development environment set up (see main README.md)

### Knowledge Requirements
- Understanding of Python programming concepts
- Familiarity with decorator patterns
- Basic knowledge of code refactoring principles
- Experience with pytest testing framework

## Quick Start Process

### Step 1: Environment Setup
```bash
# Ensure you're on the correct branch
git checkout 001-code-refactoring-modularization

# Install dependencies
pip install -r requirements.txt

# Verify environment
python -m pytest tests/ --collect-only
```

### Step 2: Run Initial Tests
```bash
# Run all tests to establish baseline
python -m pytest tests/ -v

# Run linting checks
python -m flake8 src/
python -m mypy src/
```

### Step 3: Create High Priority Utility Modules

#### 3.1 Create gemini_api.py
```bash
# Create the utility module
cat > src/gemini_api.py << 'EOF'
"""Centralized Gemini API client for all API interactions."""

import os
import requests
from typing import Dict, Any, Optional
from .json_helpers import _dumps, _parse_json_strictish

class GeminiAPIClient:
    """Centralized client for Gemini API interactions."""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required")
    
    def call_api(self, prompt: str, temperature: float = 0.1) -> str:
        """Make API call to Gemini with standardized error handling."""
        url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"
        
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "text/plain"
            }
        }
        
        try:
            response = requests.post(url, json=body, timeout=120)
            response.raise_for_status()
            return self._extract_response_text(response.json())
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}") from e
    
    def _extract_response_text(self, response_data: Dict[str, Any]) -> str:
        """Extract text from Gemini response."""
        try:
            return response_data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Invalid Gemini response: {response_data}") from e

# Global instance for easy import
gemini_client = GeminiAPIClient()
EOF
```

#### 3.2 Create search_utils.py
```bash
# Create search utilities
cat > src/search_utils.py << 'EOF'
"""Web search and parsing utilities."""

import re
from typing import List, Dict, Any
from .gemini_api import gemini_client

def search_and_parse_with_gemini(query: str, parse_prompt_template: str) -> str:
    """Perform web search and parse results using Gemini."""
    # Import here to avoid circular imports
    from .services import ddgs_search
    
    # Perform web search
    search_results = ddgs_search(query)
    
    if not search_results:
        return ""
    
    # Extract search text for parsing
    search_text = " ".join([result.get("body", "") for result in search_results[:3]])
    
    # Create parse prompt
    parse_prompt = parse_prompt_template.format(search_text=search_text)
    
    # Parse with Gemini
    try:
        return gemini_client.call_api(parse_prompt)
    except Exception:
        return ""
EOF
```

#### 3.3 Create date_utils.py
```bash
# Create date utilities
cat > src/date_utils.py << 'EOF'
"""Date formatting and handling utilities."""

import pandas as pd
from datetime import datetime
from typing import Union

def format_date_for_report(date_obj: Union[pd.Timestamp, datetime, str]) -> str:
    """Format date object for report display with multiple fallback strategies."""
    if isinstance(date_obj, pd.Timestamp):
        return date_obj.strftime("%B %d, %Y")
    elif isinstance(date_obj, datetime):
        return date_obj.strftime("%B %d, %Y")
    elif isinstance(date_obj, str):
        try:
            # Try parsing as ISO format first
            dt = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
            return dt.strftime("%B %d, %Y")
        except ValueError:
            try:
                # Try pandas parsing
                pd_date = pd.to_datetime(date_obj)
                return pd_date.strftime("%B %d, %Y")
            except Exception:
                return date_obj
    else:
        return str(date_obj)
EOF
```

### Step 4: Create Decorators

#### 4.1 Create decorators.py
```bash
# Create decorators module
cat > src/decorators.py << 'EOF'
"""Decorators for cross-cutting concerns."""

import functools
from typing import Any, Callable, Optional

def handle_api_errors(default_return: Any = None):
    """Decorator for standardized API error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"{func.__name__} failed: {e}")
                return default_return
        return wrapper
    return decorator

def with_event_reporting(event_type: str = "info"):
    """Decorator for standardized event reporting."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            on_event = kwargs.get('on_event')
            if on_event and result:
                on_event({"type": event_type, "data": result})
            return result
        return wrapper
    return decorator

def validate_input(allow_empty: bool = False):
    """Decorator for input validation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate string inputs
            for arg in args:
                if isinstance(arg, str) and not allow_empty and not arg.strip():
                    raise ValueError(f"Empty input provided to {func.__name__}")
            
            # Validate keyword arguments
            for key, value in kwargs.items():
                if isinstance(value, str) and not allow_empty and not value.strip():
                    raise ValueError(f"Empty input for {key} provided to {func.__name__}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def retry_on_parse_error(max_retries: int = 1):
    """Decorator for retry logic on parsing errors."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ValueError, KeyError, IndexError) as e:
                    last_error = e
                    if attempt == max_retries:
                        raise
            raise last_error
        return wrapper
    return decorator
EOF
```

### Step 5: Update Existing Code

#### 5.1 Update services.py to use new utilities
```bash
# Example of updating services.py to use gemini_api
# Replace repeated API call patterns with:
from .gemini_api import gemini_client

@handle_api_errors("")
def some_function():
    prompt = "Your prompt here"
    return gemini_client.call_api(prompt)
```

#### 5.2 Update orchestrator.py to use new utilities
```bash
# Example of updating orchestrator.py to use decorators
from .decorators import handle_api_errors, with_event_reporting

@handle_api_errors
@with_event_reporting("info")
def some_orchestrator_function():
    # Function implementation
    pass
```

### Step 6: Run Tests and Validation
```bash
# Run tests to ensure everything works
python -m pytest tests/ -v

# Run linting
python -m flake8 src/
python -m mypy src/

# Check for any regressions
python -m pytest tests/integration/ -v
```

### Step 7: Verify Code Reduction
```bash
# Count lines before and after refactoring
find src/ -name "*.py" -exec wc -l {} + | tail -1

# Compare with baseline (should see 35-40% reduction)
```

## Validation Steps

### 1. Functional Validation
```bash
# Run the application to ensure it works
streamlit run agentic_ticker.py

# Test with sample inputs
# - "AAPL" (stock analysis)
# - "BTC" (crypto analysis)
# - "Apple Inc. stock" (company name analysis)
```

### 2. Performance Validation
```bash
# Run performance tests if available
python -m pytest tests/performance/ -v

# Check response times manually
# Should be <200ms p95 for API operations
```

### 3. Code Quality Validation
```bash
# Check test coverage
python -m pytest tests/ --cov=src/ --cov-report=term-missing

# Ensure coverage is above 80%
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, check:
python -c "import src.gemini_api; print('OK')"
python -c "import src.decorators; print('OK')"
```

#### Test Failures
```bash
# If tests fail, run specific test files
python -m pytest tests/test_services.py -v
python -m pytest tests/test_orchestrator.py -v
```

#### Performance Issues
```bash
# If performance is degraded, check:
python -c "import time; start=time.time(); import src.gemini_api; print(f'Import time: {time.time()-start:.3f}s')"
```

## Next Steps

### 1. Complete Medium Priority Refactoring
- Create validation_utils.py
- Create chart_utils.py
- Create config.py
- Implement remaining decorators

### 2. Update Documentation
- Update AGENTS.md with new utilities
- Update README.md with refactoring changes
- Add utility module documentation

### 3. Advanced Features
- Add caching to utility functions
- Implement performance monitoring
- Add advanced error recovery

## Support

### Getting Help
- Check the main README.md for general setup
- Review the constitution.md for development standards
- Check existing test files for usage patterns
- Review the generated research.md for detailed analysis

### Reporting Issues
- Create detailed bug reports with steps to reproduce
- Include error messages and stack traces
- Provide system information and Python version
- Include test results showing the issue

---

**Note**: This quickstart guide focuses on high-priority refactoring items first. Complete all steps before proceeding to medium and low priority items.