# Troubleshooting Guide: Refactored Utility Modules

This guide provides solutions for common issues encountered when using the refactored utility modules in the Agentic Ticker codebase.

## Table of Contents

1. [Configuration Issues](#configuration-issues)
2. [Import and Module Issues](#import-and-module-issues)
3. [Decorator System Issues](#decorator-system-issues)
4. [Data Validation Issues](#data-validation-issues)
5. [Search Utilities Issues](#search-utilities-issues)
6. [Chart Utilities Issues](#chart-utilities-issues)
7. [Date Utilities Issues](#date-utilities-issues)
8. [Performance Issues](#performance-issues)
9. [Migration Issues](#migration-issues)
10. [Testing Issues](#testing-issues)

---

## Configuration Issues

### Issue: Configuration Not Loading

**Symptoms:**
- `GEMINI_API_KEY` not found
- Configuration values are empty
- Application fails to start

**Solutions:**

```python
# Check environment variables
import os
print(f"GEMINI_API_KEY present: {'GEMINI_API_KEY' in os.environ}")

# Explicitly load configuration
from src.config import load_config
config = load_config("config.json")  # Specify config file explicitly

# Check configuration validation
errors = config.validate()
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")

# Set environment variables if missing
os.environ['GEMINI_API_KEY'] = 'your-actual-api-key'
os.environ['LOG_LEVEL'] = 'INFO'
```

### Issue: YAML Configuration Not Working

**Symptoms:**
- `ImportError: No module named 'yaml'`
- YAML config files not loading

**Solutions:**

```bash
# Install PyYAML
pip install PyYAML

# Or use JSON configuration instead
# Rename config.yaml to config.json and use JSON format
```

### Issue: Hot Reload Not Working

**Symptoms:**
- Configuration changes not reflected
- Hot reload interval not respected

**Solutions:**

```python
from src.config import load_config

# Enable hot reload with specific interval
config = load_config()
config.hot_reload_enabled = True
config.hot_reload_interval = 30  # seconds

# Manual reload
from src.config import reload_config
reload_config()
```

---

## Import and Module Issues

### Issue: Import Errors with Utility Modules

**Symptoms:**
- `ModuleNotFoundError: No module named 'src.chart_utils'`
- Import errors in scripts

**Solutions:**

```python
# Add src to Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Or use absolute imports
from src.chart_utils import create_price_traces
from src.decorators import handle_errors, log_execution
from src.validation_utils import validate_dataframe

# Check if modules exist
import os
print(f"chart_utils.py exists: {os.path.exists('src/chart_utils.py')}")
```

### Issue: Circular Import Errors

**Symptoms:**
- `ImportError: cannot import name 'X' from partially initialized module 'Y'`
- Circular dependency errors

**Solutions:**

```python
# Use lazy imports inside functions
def my_function():
    from src.chart_utils import create_price_traces
    # Use the import here
    
# Or restructure imports to avoid circular dependencies
# Move shared utilities to separate modules
```

### Issue: Module Not Found in Jupyter Notebooks

**Symptoms:**
- Imports work in scripts but not in Jupyter notebooks

**Solutions:**

```python
# In Jupyter notebook, add to first cell:
import sys
import os
sys.path.append(os.path.abspath('src'))

# Or install as editable package
# pip install -e .
```

---

## Decorator System Issues

### Issue: Decorators Not Working

**Symptoms:**
- Functions not being decorated
- No logging or error handling

**Solutions:**

```python
# Check decorator order - order matters!
@handle_errors(default_return={})
@log_execution(include_args=True)
@validate_inputs(ticker='non_empty_string')
def my_function(ticker: str):
    # Function implementation
    pass

# Ensure proper decorator syntax
from src.decorators import handle_errors, log_execution, validate_inputs

# Test decorators individually
@handle_errors(default_return="error")
def test_function():
    raise ValueError("Test error")

result = test_function()
print(f"Result: {result}")  # Should return "error"
```

### Issue: Validation Errors

**Symptoms:**
- `TypeError: Parameter 'X' must be a pandas DataFrame`
- Validation failing for correct inputs

**Solutions:**

```python
# Check parameter names match
@validate_inputs(data='dataframe', threshold='positive_number')
def process_data(data: pd.DataFrame, threshold: float):
    pass

# Use correct validator types
validators = {
    'dataframe': 'dataframe',
    'list_of_dicts': 'list_of_dicts',
    'non_empty_list': 'non_empty_list',
    'positive_number': 'positive_number',
    'non_empty_string': 'non_empty_string'
}

# Custom validation function
def custom_validator(value):
    return value > 0 and value < 100

@validate_inputs(score=custom_validator)
def process_score(score: float):
    pass
```

### Issue: Caching Not Working

**Symptoms:**
- Cache not hitting
- Functions executing repeatedly

**Solutions:**

```python
# Check cache key generation
@cache_result(max_size=128)
def expensive_function(arg1, arg2):
    # Function must have hashable arguments
    return result

# Use hashable arguments
# Lists and dicts won't work as cache keys
# Convert to tuples and frozensets

# Clear cache if needed
import functools
func.cache_clear()  # If using functools.lru_cache
```

---

## Data Validation Issues

### Issue: DataFrame Validation Failing

**Symptoms:**
- `Missing required columns: ['X', 'Y']`
- `Column 'Z' should be numeric but is object`

**Solutions:**

```python
from src.validation_utils import validate_dataframe

# Check actual DataFrame structure
print("DataFrame columns:", df.columns.tolist())
print("DataFrame dtypes:")
print(df.dtypes)

# Validate with correct parameters
is_valid = validate_dataframe(
    df,
    required_columns=['date', 'close', 'volume'],
    column_types={
        'date': 'datetime',
        'close': 'numeric',
        'volume': 'numeric'
    },
    min_rows=10
)

# Fix data types if needed
df['date'] = pd.to_datetime(df['date'])
df['close'] = pd.to_numeric(df['close'])
df['volume'] = pd.to_numeric(df['volume'])
```

### Issue: Numeric Data Cleaning Problems

**Symptoms:**
- `TypeError: unsupported operand type(s)`
- NaN values not being handled

**Solutions:**

```python
from src.validation_utils import clean_numeric_data

# Check for problematic values
print("NaN count:", df['price'].isna().sum())
print("None count:", (df['price'] == None).sum())
print("Data type:", df['price'].dtype)

# Clean with appropriate default value
cleaned = clean_numeric_data(
    df['price'],
    default_value=0.0,
    remove_outliers=True,
    outlier_threshold=2.5
)

# Handle different NaN representations
df['price'] = df['price'].replace([None, 'NaN', ''], np.nan)
df['price'] = clean_numeric_data(df['price'])
```

### Issue: String Data Cleaning Issues

**Symptoms:**
- Whitespace not being removed
- Case conversion not working

**Solutions:**

```python
from src.validation_utils import clean_string_data

# Check original data
print("Original:", repr(df['company'].iloc[0]))

# Clean with specific options
cleaned = clean_string_data(
    df['company'],
    default_value="Unknown",
    strip_whitespace=True,
    convert_to_upper=True
)

print("Cleaned:", repr(cleaned.iloc[0]))

# Handle mixed types
df['company'] = df['company'].astype(str)
cleaned = clean_string_data(df['company'])
```

---

## Search Utilities Issues

### Issue: DDGS Search Not Working

**Symptoms:**
- `ImportError: No module named 'ddgs'`
- Search returning empty results
- Connection timeouts

**Solutions:**

```bash
# Install DDGS
pip install ddgs

# Or use fallback search
from src.search_utils import SearchConfig
config = SearchConfig(enable_fallback=True)
```

### Issue: Gemini API Calls Failing

**Symptoms:**
- `SearchError: Gemini API key not configured`
- API rate limits exceeded
- Invalid responses

**Solutions:**

```python
from src.search_utils import SearchUtils

# Check API key configuration
import os
print(f"GEMINI_API_KEY set: {'GEMINI_API_KEY' in os.environ}")

# Set API key if missing
os.environ['GEMINI_API_KEY'] = 'your-actual-api-key'

# Use with proper error handling
search_utils = SearchUtils()
try:
    ticker = search_utils.parse_ticker_from_search(results, "Apple Inc")
except Exception as e:
    print(f"Ticker parsing failed: {e}")
    ticker = "AAPL"  # Fallback
```

### Issue: Search Results Empty

**Symptoms:**
- `web_search()` returning empty list
- No results for valid queries

**Solutions:**

```python
from src.search_utils import SearchUtils, SearchConfig

# Increase timeout and retry count
config = SearchConfig(
    timeout=60,
    retry_count=3,
    max_results=10
)

search_utils = SearchUtils(config)

# Try different query formats
queries = [
    "Apple Inc stock",
    "AAPL ticker symbol",
    "Apple stock price"
]

for query in queries:
    results = search_utils.web_search(query)
    if results:
        print(f"Success with query: {query}")
        break
```

---

## Chart Utilities Issues

### Issue: Plotly Charts Not Displaying

**Symptoms:**
- Charts not showing in Jupyter
- `ImportError: No module named 'plotly'`

**Solutions:**

```bash
# Install Plotly
pip install plotly

# In Jupyter, enable offline mode
import plotly.io as pio
pio.renderers.default = 'notebook'

# Or use browser display
pio.renderers.default = 'browser'
```

### Issue: Animation Frames Not Working

**Symptoms:**
- Animation controls not appearing
- Frames not playing

**Solutions:**

```python
from src.chart_utils import create_animation_frames, create_animation_controls

# Check data format
print("DataFrame shape:", df.shape)
print("Required columns present:", all(col in df.columns for col in ['date', 'close']))

# Create frames with proper function
def create_traces(current_df):
    if current_df.empty:
        return []
    return [go.Scatter(
        x=current_df['date'],
        y=current_df['close'],
        mode='lines+markers'
    )]

frames = create_animation_frames(df, create_traces)
print(f"Created {len(frames)} frames")

# Add animation controls
animation_controls = create_animation_controls(duration=500)
fig.update_layout(**animation_controls)
```

### Issue: Color Utilities Returning Invalid Colors

**Symptoms:**
- Invalid hex color format
- RGBA conversion failing

**Solutions:**

```python
from src.chart_utils import get_trend_color, hex_to_rgba

# Validate hex color format
hex_color = "#1f77b4"
if not hex_color.startswith('#') or len(hex_color) != 7:
    hex_color = "#1f77b4"  # Default color

# Convert to RGBA
rgba = hex_to_rgba(hex_color, alpha=0.3)

# Get trend color with valid inputs
trend = "UP"  # Valid values: "UP", "DOWN", "NEUTRAL"
confidence = 0.8  # Valid range: 0.0 - 1.0
color = get_trend_color(trend, confidence)
```

---

## Date Utilities Issues

### Issue: Date Conversion Failing

**Symptoms:**
- `Error converting to datetime`
- Invalid date formats

**Solutions:**

```python
from src.date_utils import safe_to_datetime

# Check original date format
print("Sample dates:", df['date'].head())
print("Date dtype:", df['date'].dtype)

# Try different date formats
date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']

for fmt in date_formats:
    try:
        df['date'] = safe_to_datetime(df['date'], date_format=fmt)
        print(f"Success with format: {fmt}")
        break
    except Exception as e:
        print(f"Failed with format {fmt}: {e}")

# Use coerce to handle invalid dates
df['date'] = safe_to_datetime(df['date'], errors='coerce')
print("NaN dates after conversion:", df['date'].isna().sum())
```

### Issue: Date Range Calculation Wrong

**Symptoms:**
- Incorrect date range returned
- Missing dates not detected

**Solutions:**

```python
from src.date_utils import get_date_range, get_missing_dates

# Check for NaN dates
print("NaN dates:", df['date'].isna().sum())

# Remove NaN dates before processing
df_clean = df.dropna(subset=['date'])

# Get date range
date_range = get_date_range(df_clean, 'date')
if date_range:
    print(f"Date range: {date_range[0]} to {date_range[1]}")

# Find missing dates
missing = get_missing_dates(df_clean, 'date', freq='D')
print(f"Missing {len(missing)} dates")
```

### Issue: Date Filtering Not Working

**Symptoms:**
- Filtered DataFrame empty
- Date range not applied correctly

**Solutions:**

```python
from src.date_utils import filter_by_date_range
from datetime import datetime

# Ensure proper date format
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

# Check date column exists and is datetime
print("Date column exists:", 'date' in df.columns)
print("Date dtype:", df['date'].dtype)

# Apply filter
filtered_df = filter_by_date_range(
    df,
    start_date=start_date,
    end_date=end_date,
    date_column='date'
)

print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(filtered_df)}")
```

---

## Performance Issues

### Issue: Slow Function Execution

**Symptoms:**
- Functions taking too long to execute
- Timeouts occurring

**Solutions:**

```python
from src.decorators import time_execution, cache_result

# Identify slow functions
@time_execution(log_threshold=1.0)
def slow_function():
    # Function implementation
    pass

# Add caching for expensive operations
@cache_result(max_size=128)
def expensive_computation(data):
    # Expensive operation
    return result

# Optimize data processing
# Use vectorized operations instead of loops
# Process data in chunks for large datasets
```

### Issue: High Memory Usage

**Symptoms:**
- Memory errors
- System running out of memory

**Solutions:**

```python
# Process data in chunks
def process_large_dataset(df, chunk_size=1000):
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    return pd.concat(results)

# Use generators for large data
def data_generator():
    for item in large_dataset:
        yield process_item(item)

# Clear unused variables
del large_dataframe
import gc
gc.collect()
```

### Issue: Cache Memory Issues

**Symptoms:**
- Cache using too much memory
- Cache not clearing properly

**Solutions:**

```python
from src.decorators import cache_result

# Reduce cache size
@cache_result(max_size=32)  # Smaller cache
@cache_result(max_size=16)  # Even smaller

def my_function():
    pass

# Clear cache manually
# For functools.lru_cache:
my_function.cache_clear()

# For custom caching, implement cache clearing
```

---

## Migration Issues

### Issue: Compatibility Warnings Not Showing

**Symptoms:**
- No warnings for deprecated functions
- Old code still working without warnings

**Solutions:**

```python
from src.compatibility_wrappers import enable_compatibility_warnings

# Enable warnings programmatically
enable_compatibility_warnings()

# Or set environment variable
import os
os.environ['COMPATIBILITY_WARNINGS'] = 'true'

# Check compatibility status
from src.compatibility_wrappers import get_compatibility_status
status = get_compatibility_status()
print(f"Warnings enabled: {status.get('warnings_enabled', False)}")
```

### Issue: Old Functions Not Working After Migration

**Symptoms:**
- Import errors for old function names
- Functions not found

**Solutions:**

```python
# Check available functions
from src.compatibility_wrappers import get_compatibility_status
status = get_compatibility_status()
print("Available mappings:", status.get('registered_mappings', {}))

# Use new function names
# Old: from src.services import validate_ticker
# New: from src.search_utils import SearchUtils

search_utils = SearchUtils()
ticker = search_utils.validate_and_clean_ticker("AAPL")
```

### Issue: Strict Mode Too Restrictive

**Symptoms:**
- Exceptions being raised for deprecated functions
- Application breaking during migration

**Solutions:**

```python
from src.compatibility_wrappers import set_strict_mode

# Disable strict mode during migration
set_strict_mode(False)

# Or set via environment variable
os.environ['COMPATIBILITY_STRICT'] = 'false'

# Gradually enable strict mode for specific modules
# Migrate one module at a time
```

---

## Testing Issues

### Issue: Tests Not Finding Modules

**Symptoms:**
- `ModuleNotFoundError` in tests
- Import errors during testing

**Solutions:**

```python
# In conftest.py or test files
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Or use pytest configuration
# Add to pytest.ini or pyproject.toml:
# [tool.pytest.ini_options]
# pythonpath = ["src"]
```

### Issue: Mock Environment Variables Not Working

**Symptoms:**
- Tests failing due to missing API keys
- Configuration not using test values

**Solutions:**

```python
# In conftest.py
import os

# Set test environment variables
os.environ['GEMINI_API_KEY'] = 'test_api_key'
os.environ['GEMINI_MODEL'] = 'test_model'

# Or use pytest fixtures
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test_key')
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
```

### Issue: Test Data Fixtures Not Loading

**Symptoms:**
- Test data not available
- Fixtures not being used

**Solutions:**

```python
# In conftest.py
import pytest
import pandas as pd

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'price': [100, 101, 102, 103, 104]
    })

# Use fixture in tests
def test_function(sample_data):
    assert len(sample_data) == 5
    assert 'price' in sample_data.columns
```

---

## General Debugging Tips

### Enable Detailed Logging

```python
import logging
from src.config import setup_logging, load_config

# Load configuration with debug logging
config = load_config()
config.logging.level = "DEBUG"
setup_logging(config)

# Or set manually
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

### Check Python Path

```python
import sys
import os

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print(f"Current directory: {os.getcwd()}")
print(f"src directory exists: {os.path.exists('src')}")
```

### Verify Dependencies

```bash
# Check installed packages
pip list | grep -E "(pandas|plotly|yfinance|pycoingecko|ddgs)"

# Install missing dependencies
pip install -r requirements.txt

# Check for version conflicts
pip check
```

### Test Individual Components

```python
# Test configuration
from src.config import load_config
config = load_config()
print("Config loaded successfully")

# Test decorators
from src.decorators import handle_errors
@handle_errors(default_return="test")
def test_decorator():
    raise ValueError("Test")
result = test_decorator()
print(f"Decorator result: {result}")

# Test utilities
from src.validation_utils import validate_dataframe
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3]})
is_valid = validate_dataframe(df)
print(f"Validation result: {is_valid}")
```

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs** - Enable debug logging to see detailed error messages
2. **Verify your environment** - Ensure all dependencies are installed and environment variables are set
3. **Test components individually** - Isolate the problematic component and test it separately
4. **Check the documentation** - Review the API documentation for correct usage
5. **Review examples** - Look at the usage examples for similar use cases

For additional support:
- Check the project documentation in the `docs/` directory
- Review the test files for working examples
- Examine the source code for implementation details
- Open an issue on the project repository with detailed error information