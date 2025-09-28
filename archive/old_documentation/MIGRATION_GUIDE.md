# Migration Guide: Backward Compatibility Layer

This guide provides instructions for migrating from the existing function-based architecture to the new utility modules architecture.

## Overview

The backward compatibility layer ensures that existing code continues to work without modification while providing a smooth migration path to the new modular architecture.

## What's Changing?

### Old Architecture (Services Module)
- All functions were in a single `services.py` module
- Functions were tightly coupled and hard to test individually
- Limited reusability and modularity

### New Architecture (Utility Modules)
- Functions are organized into logical modules:
  - `utility_modules/validation.py` - Asset validation functions
  - `utility_modules/data_loading.py` - Data fetching functions
  - `utility_modules/analysis.py` - Technical analysis functions
  - `utility_modules/reporting.py` - Report generation functions
  - `utility_modules/search.py` - Search functionality
  - `utility_modules/company_info.py` - Company information
  - `utility_modules/crypto_info.py` - Cryptocurrency information
  - `utility_modules/classification.py` - Asset type classification

## Migration Steps

### Step 1: Enable Compatibility Warnings

Before migrating, enable compatibility warnings to identify all deprecated function usage:

```python
from src.compatibility_wrappers import enable_compatibility_warnings
enable_compatibility_warnings()
```

### Step 2: Update Imports

Replace old imports with new ones:

#### Before (Old)
```python
from src.services import (
    validate_ticker,
    get_company_info,
    get_crypto_info,
    load_prices,
    compute_indicators,
    detect_events,
    forecast_prices,
    build_report
)
```

#### After (New)
```python
from src.utility_modules.validation import validate_ticker
from src.utility_modules.company_info import get_company_info
from src.utility_modules.crypto_info import get_crypto_info
from src.utility_modules.data_loading import load_prices, load_crypto_prices
from src.utility_modules.analysis import compute_indicators, detect_events, forecast_prices
from src.utility_modules.reporting import build_report
from src.utility_modules.search import ddgs_search
from src.utility_modules.classification import classify_asset_type
```

### Step 3: Update Function Calls

Most function signatures remain the same, but some have been enhanced:

#### validate_ticker
```python
# Old usage (still works with compatibility layer)
result = validate_ticker("AAPL")

# New usage (recommended)
result = validate_ticker("AAPL", asset_type="stock")  # Optional asset_type parameter
```

#### get_crypto_info
```python
# Old usage (still works)
info = get_crypto_info("BTC")

# New usage (enhanced)
info = get_crypto_info("BTC", include_market_data=True)  # Additional parameters
```

#### load_prices / load_crypto_prices
```python
# Old usage (still works)
prices = load_prices("AAPL", 30)

# New usage (enhanced)
prices = load_prices("AAPL", days=30, include_volume=True)  # Named parameters
```

### Step 4: Handle Breaking Changes

Some functions have been improved with breaking changes:

#### build_report
```python
# Old usage
report = build_report("AAPL", events, forecasts, company_info, crypto_info)

# New usage (structured parameters)
report = build_report(
    ticker="AAPL",
    events=events,
    forecasts=forecasts,
    company_info=company_info,
    crypto_info=crypto_info,
    template="detailed"  # New parameter for report template
)
```

#### compute_indicators
```python
# Old usage
indicators = compute_indicators(price_data)

# New usage (configurable indicators)
indicators = compute_indicators(
    price_data,
    indicators=["ma5", "ma10", "rsi", "macd"],  # Specify which indicators to compute
    window_sizes={"ma5": 5, "ma10": 10}  # Custom window sizes
)
```

### Step 5: Remove Compatibility Layer

Once all code has been migrated, you can disable the compatibility layer:

```python
from src.compatibility_wrappers import disable_compatibility_layer
disable_compatibility_layer()
```

Or via environment variable:
```bash
export COMPATIBILITY_ENABLED=false
```

## Configuration Options

### Environment Variables

Configure the compatibility layer using environment variables:

```bash
# Enable/disable compatibility layer
export COMPATIBILITY_ENABLED=true

# Show/hide deprecation warnings
export COMPATIBILITY_WARNINGS=true

# Enable strict mode (raises exceptions instead of warnings)
export COMPATIBILITY_STRICT=false

# Enable fallback to legacy implementations
export COMPATIBILITY_FALLBACK=true

# Set migration deadline
export COMPATIBILITY_DEADLINE=2025-12-31
```

### Programmatic Configuration

```python
from src.compatibility_wrappers import (
    enable_compatibility_layer,
    disable_compatibility_layer,
    set_strict_mode,
    set_migration_deadline
)

# Enable compatibility layer
enable_compatibility_layer()

# Set strict mode (raises exceptions for deprecated functions)
set_strict_mode(True)

# Set migration deadline
from datetime import datetime
set_migration_deadline(datetime(2025, 12, 31))
```

## Testing Migration

### 1. Run Tests with Compatibility Layer

```bash
# Run tests with compatibility warnings enabled
export COMPATIBILITY_WARNINGS=true
python -m pytest tests/

# Run tests in strict mode
export COMPATIBILITY_STRICT=true
python -m pytest tests/
```

### 2. Check for Deprecated Usage

```python
from src.compatibility import get_compatibility_status

status = get_compatibility_status()
print(f"Compatibility enabled: {status['enabled']}")
print(f"Days until deadline: {status['days_until_deadline']}")
print(f"Validation issues: {status['validation_issues']}")
```

### 3. Gradual Migration

Migrate module by module:

```python
# Step 1: Migrate validation functions
from src.utility_modules.validation import validate_ticker

# Step 2: Migrate data loading functions
from src.utility_modules.data_loading import load_prices, load_crypto_prices

# Step 3: Migrate analysis functions
from src.utility_modules.analysis import compute_indicators, detect_events, forecast_prices

# Continue with remaining modules...
```

## Benefits of Migration

### 1. Better Modularity
- Functions are organized by purpose
- Easier to find and maintain specific functionality
- Reduced coupling between modules

### 2. Improved Testing
- Each module can be tested independently
- Better test coverage and isolation
- Easier to mock dependencies

### 3. Enhanced Performance
- Lazy loading of modules
- Reduced memory footprint
- Faster import times

### 4. Better Documentation
- Each module has focused documentation
- Clearer API boundaries
- Easier to understand and maintain

## Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# If you get import errors, try:
import sys
sys.path.append('path/to/agentic-ticker')
```

#### 2. Missing Functions
```python
# If a function is missing, check if it's been renamed:
from src.compatibility_wrappers import get_compatibility_status
status = get_compatibility_status()
print("Available mappings:", status['registered_mappings'])
```

#### 3. Type Errors
```python
# If you encounter type errors, ensure you're using the correct function signatures:
# Check the new module's docstring for updated parameter types
```

### Getting Help

1. **Check Compatibility Status**: Use `get_compatibility_status()` to see current state
2. **Enable Warnings**: Turn on compatibility warnings to see deprecated usage
3. **Consult Migration Guide**: Refer to this guide for step-by-step instructions
4. **Run Tests**: Ensure all tests pass after migration
5. **Check Documentation**: Review new module documentation for updated APIs

## Timeline

- **Phase 1 (Current)**: Compatibility layer enabled, warnings shown
- **Phase 2 (1 month)**: Migration guide published, new modules stable
- **Phase 3 (3 months)**: Compatibility warnings become errors in strict mode
- **Phase 4 (6 months)**: Compatibility layer deprecated but still available
- **Phase 5 (12 months)**: Compatibility layer removed

## Conclusion

The backward compatibility layer ensures a smooth transition to the new modular architecture. By following this migration guide, you can gradually update your code while maintaining full functionality throughout the process.

For additional support or questions, please refer to the project documentation or open an issue on the project repository.