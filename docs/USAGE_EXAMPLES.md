# Usage Examples: Refactored Utility Modules

This document provides practical examples of how to use the new utility modules in the refactored Agentic Ticker codebase.

## Table of Contents

1. [Configuration System Examples](#configuration-system-examples)
2. [Decorator System Examples](#decorator-system-examples)
3. [Chart Utilities Examples](#chart-utilities-examples)
4. [Date Utilities Examples](#date-utilities-examples)
5. [Validation Utilities Examples](#validation-utilities-examples)
6. [Search Utilities Examples](#search-utilities-examples)
7. [JSON Helpers Examples](#json-helpers-examples)
8. [Complete Workflow Examples](#complete-workflow-examples)
9. [Migration Examples](#migration-examples)

---

## Configuration System Examples

### Basic Configuration Setup

```python
from src.config import load_config, setup_logging

# Load configuration (automatically loads from file or env vars)
config = load_config()

# Set up logging based on configuration
setup_logging()

# Use configuration values
print(f"Gemini API Key: {config.gemini.api_key}")
print(f"Default analysis days: {config.analysis.default_days}")
print(f"Web search enabled: {config.feature_flags.enable_web_search}")
```

### Custom Configuration File

```python
from src.config import load_config, AppConfig

# Load from specific configuration file
config = load_config("my_config.json")

# Or create custom configuration programmatically
config = AppConfig()
config.gemini.api_key = "your-api-key"
config.analysis.default_days = 60
config.feature_flags.enable_web_search = False

# Save configuration to file
config.save_to_file("custom_config.json")
```

### Configuration Validation

```python
from src.config import load_config

config = load_config()

# Validate configuration
errors = config.validate()
if errors:
    print("Configuration errors found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid!")
```

### Environment Variable Configuration

```bash
# Set environment variables
export GEMINI_API_KEY="your-actual-api-key"
export GEMINI_MODEL="gemini-2.5-flash-lite"
export LOG_LEVEL="DEBUG"
export ENABLE_WEB_SEARCH="true"
export ENABLE_CRYPTO_ANALYSIS="true"
export DEFAULT_ANALYSIS_DAYS="45"

# Run application
python agentic_ticker.py
```

---

## Decorator System Examples

### Basic Error Handling

```python
from src.decorators import handle_errors

@handle_errors(default_return=[], log_errors=True)
def fetch_stock_data(ticker: str):
    """Fetch stock data with automatic error handling."""
    # This might raise an exception
    data = yf.download(ticker, period="1mo")
    return data.to_dict('records')

# Usage - returns empty list on error instead of crashing
result = fetch_stock_data("INVALID_TICKER")
print(f"Result: {result}")  # Returns [] on error
```

### Execution Logging

```python
from src.decorators import log_execution

@log_execution(include_args=True, include_result=False)
def process_financial_data(data: dict):
    """Process data with execution logging."""
    # Function execution will be logged
    processed = {k: v * 2 for k, v in data.items()}
    return processed

# Usage - execution will be logged
result = process_financial_data({'price': 100, 'volume': 1000})
```

### Performance Timing

```python
from src.decorators import time_execution

@time_execution(log_threshold=0.1)  # Log if execution takes > 0.1 seconds
def slow_calculation(data: list):
    """Perform calculation with timing monitoring."""
    import time
    time.sleep(0.2)  # Simulate slow operation
    return sum(data) / len(data)

# Usage - execution time will be logged
result = slow_calculation([1, 2, 3, 4, 5])
```

### Input Validation

```python
from src.decorators import validate_inputs
import pandas as pd

@validate_inputs(
    prices='dataframe',
    window='positive_number',
    ticker='non_empty_string'
)
def calculate_moving_average(prices: pd.DataFrame, window: int, ticker: str):
    """Calculate moving average with input validation."""
    return prices['close'].rolling(window=window).mean()

# Usage - inputs will be validated
df = pd.DataFrame({'close': [100, 101, 102, 103, 104]})
ma = calculate_moving_average(df, 3, "AAPL")  # Valid inputs

# This will raise validation error:
# ma = calculate_moving_average("not_a_dataframe", -1, "")
```

### Caching Results

```python
from src.decorators import cache_result
import yfinance as yf

@cache_result(max_size=32)
def get_company_info(ticker: str):
    """Get company info with caching for repeated calls."""
    print(f"Fetching info for {ticker}...")  # This will only print on cache miss
    ticker = yf.Ticker(ticker)
    return ticker.info

# Usage - first call fetches data, subsequent calls use cache
info1 = get_company_info("AAPL")
info2 = get_company_info("AAPL")  # Uses cache - no fetch
info3 = get_company_info("MSFT")  # Fetches new data
```

### Retry Logic

```python
from src.decorators import retry_on_failure
import requests

@retry_on_failure(max_attempts=3, delay=1.0, exceptions=(requests.RequestException,))
def fetch_web_data(url: str):
    """Fetch web data with retry logic for network failures."""
    response = requests.get(url, timeout=10)
    return response.json()

# Usage - will retry up to 3 times on network errors
data = fetch_web_data("https://api.example.com/data")
```

### Combined Decorators

```python
from src.decorators import (
    handle_errors, log_execution, time_execution, 
    validate_inputs, cache_result, retry_on_failure
)

@handle_errors(default_return={})
@log_execution(include_args=True)
@time_execution(log_threshold=0.5)
@validate_inputs(ticker='non_empty_string', days='positive_number')
@cache_result(max_size=64)
@retry_on_failure(max_attempts=3, delay=0.5)
def get_stock_analysis(ticker: str, days: int) -> dict:
    """
    Comprehensive stock analysis with full decorator coverage.
    
    - Error handling with fallback
    - Execution logging
    - Performance timing
    - Input validation
    - Result caching
    - Retry logic
    """
    # Fetch and analyze stock data
    import yfinance as yf
    
    data = yf.download(ticker, period=f"{days}d")
    
    analysis = {
        'ticker': ticker,
        'current_price': data['Close'].iloc[-1],
        'avg_volume': data['Volume'].mean(),
        'price_change': data['Close'].pct_change().iloc[-1],
        'volatility': data['Close'].std()
    }
    
    return analysis

# Usage - all decorators will be applied
result = get_stock_analysis("AAPL", 30)
print(f"Analysis result: {result}")
```

---

## Chart Utilities Examples

### Basic Chart Creation

```python
import pandas as pd
import plotly.graph_objects as go
from src.chart_utils import create_price_traces, create_chart_layout

# Sample data
dates = pd.date_range('2023-01-01', periods=30)
prices = [100 + i * 2 + (i % 5) for i in range(30)]
df = pd.DataFrame({'date': dates, 'close': prices})

# Create price traces
price_traces = create_price_traces(df)

# Create chart layout
layout = create_chart_layout(
    title="Stock Price Analysis",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=500
)

# Create figure
fig = go.Figure(data=price_traces, layout=layout)
fig.show()
```

### Chart with Indicators

```python
import pandas as pd
import plotly.graph_objects as go
from src.chart_utils import create_price_traces, create_chart_layout

# Sample data with indicators
dates = pd.date_range('2023-01-01', periods=30)
prices = [100 + i * 2 + (i % 5) for i in range(30)]
ma5 = [sum(prices[max(0, i-4):i+1])/len(prices[max(0, i-4):i+1]) for i in range(30)]
ma10 = [sum(prices[max(0, i-9):i+1])/len(prices[max(0, i-9):i+1]) for i in range(30)]

price_df = pd.DataFrame({'date': dates, 'close': prices})
ind_df = pd.DataFrame({'date': dates, 'ma5': ma5, 'ma10': ma10})

# Create traces with indicators
price_traces = create_price_traces(price_df, ind_df)

# Create chart
fig = go.Figure(data=price_traces)
fig.update_layout(title="Stock Price with Moving Averages")
fig.show()
```

### Animated Charts

```python
import pandas as pd
import plotly.graph_objects as go
from src.chart_utils import (
    create_animation_frames, create_animation_controls, 
    create_chart_layout, create_price_traces
)

# Sample data for animation
dates = pd.date_range('2023-01-01', periods=30)
prices = [100 + i * 2 + (i % 5) for i in range(30)]
df = pd.DataFrame({'date': dates, 'close': prices})

# Create animation frames
def create_traces(current_df):
    return [go.Scatter(
        x=current_df['date'],
        y=current_df['close'],
        mode='lines+markers',
        name='Price'
    )]

frames = create_animation_frames(df, create_traces)

# Create animation controls
animation_controls = create_animation_controls(
    duration=500,
    transition_duration=300
)

# Create initial figure
fig = go.Figure(
    data=create_traces(df.iloc[:1]),
    frames=frames,
    layout=go.Layout(**animation_controls)
)

fig.update_layout(title="Animated Stock Price Chart")
fig.show()
```

### Forecast Charts with Confidence Bands

```python
import pandas as pd
import plotly.graph_objects as go
from src.chart_utils import create_forecast_traces, hex_to_rgba

# Sample forecast data
dates = pd.date_range('2023-02-01', periods=5)
forecast_prices = [110, 112, 115, 117, 120]
confidence = [0.8, 0.75, 0.7, 0.65, 0.6]

forecast_df = pd.DataFrame({
    'date': dates,
    'forecast_price': forecast_prices,
    'confidence': confidence
})

# Create forecast traces
forecast_traces = create_forecast_traces(forecast_df, line_color='#2ca02c')

# Create figure
fig = go.Figure(data=forecast_traces)
fig.update_layout(
    title="Price Forecast with Confidence Band",
    xaxis_title="Date",
    yaxis_title="Price ($)"
)
fig.show()
```

### Color Utilities

```python
from src.chart_utils import get_trend_color, hex_to_rgba

# Get trend colors
up_color = get_trend_color('UP', 0.8)      # Dark green (high confidence)
up_color_weak = get_trend_color('UP', 0.4)  # Light green (low confidence)
down_color = get_trend_color('DOWN', 0.8)   # Dark red (high confidence)
neutral_color = get_trend_color('NEUTRAL', 0.6)  # Orange

print(f"Strong up trend: {up_color}")
print(f"Weak up trend: {up_color_weak}")
print(f"Strong down trend: {down_color}")
print(f"Neutral trend: {neutral_color}")

# Convert to RGBA with transparency
rgba_color = hex_to_rgba('#1f77b4', alpha=0.3)
print(f"RGBA color: {rgba_color}")  # rgba(31, 119, 180, 0.3)
```

---

## Date Utilities Examples

### Date Conversion and Validation

```python
import pandas as pd
from src.date_utils import safe_to_datetime, validate_date_column

# Sample data with mixed date formats
df = pd.DataFrame({
    'date': ['2023-01-01', '2023/01/02', '2023-01-03', 'invalid_date'],
    'price': [100, 101, 102, 103]
})

# Safe date conversion
df['date'] = safe_to_datetime(df['date'])
print(df['date'])

# Validate date column
is_valid = validate_date_column(df, 'date')
print(f"Date column valid: {is_valid}")
```

### Date Range Operations

```python
import pandas as pd
from src.date_utils import get_date_range, filter_by_date_range
from datetime import datetime

# Sample time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
prices = [100 + i * 0.5 for i in range(100)]
df = pd.DataFrame({'date': dates, 'price': prices})

# Get date range
date_range = get_date_range(df, 'date')
if date_range:
    start_date, end_date = date_range
    print(f"Data spans from {start_date} to {end_date}")

# Filter by date range
filtered_df = filter_by_date_range(
    df,
    start_date=datetime(2023, 2, 1),
    end_date=datetime(2023, 2, 28),
    date_column='date'
)

print(f"Filtered data has {len(filtered_df)} rows")
```

### Date Component Extraction

```python
import pandas as pd
from src.date_utils import add_date_components

# Sample data
dates = pd.date_range('2023-01-01', periods=10, freq='D')
df = pd.DataFrame({'date': dates, 'price': range(100, 110)})

# Add date components
enhanced_df = add_date_components(df, 'date')

print("Enhanced DataFrame columns:")
print(enhanced_df.columns.tolist())
# Output: ['date', 'price', 'date_year', 'date_month', 'date_day', 'date_dayofweek', 'date_quarter']

print("\nSample data with date components:")
print(enhanced_df[['date', 'date_year', 'date_month', 'date_dayofweek']].head())
```

### Missing Date Detection

```python
import pandas as pd
from src.date_utils import get_missing_dates

# Create data with missing dates
dates = ['2023-01-01', '2023-01-02', '2023-01-04', '2023-01-05', '2023-01-07']
prices = [100, 101, 103, 104, 106]
df = pd.DataFrame({'date': pd.to_datetime(dates), 'price': prices})

# Find missing dates
missing_dates = get_missing_dates(df, 'date', freq='D')

print(f"Missing dates: {missing_dates}")
# Output: [Timestamp('2023-01-03 00:00:00'), Timestamp('2023-01-06 00:00:00')]
```

---

## Validation Utilities Examples

### DataFrame Validation

```python
import pandas as pd
from src.validation_utils import validate_dataframe

# Sample financial data
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
    'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
    'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
})

# Basic validation
is_valid = validate_dataframe(df)
print(f"DataFrame valid: {is_valid}")

# Advanced validation with requirements
is_valid = validate_dataframe(
    df,
    required_columns=['date', 'open', 'high', 'low', 'close', 'volume'],
    min_rows=5,
    max_rows=1000,
    column_types={
        'date': 'datetime',
        'open': 'numeric',
        'high': 'numeric',
        'low': 'numeric',
        'close': 'numeric',
        'volume': 'numeric'
    }
)
print(f"DataFrame meets requirements: {is_valid}")
```

### Data Cleaning

```python
import pandas as pd
import numpy as np
from src.validation_utils import clean_numeric_data, clean_string_data

# Sample data with issues
df = pd.DataFrame({
    'price': [100, 101, None, 103, np.nan, 105, 1000],  # Has None, NaN, and outlier
    'company': ['Apple', 'Microsoft', None, 'Google', '', 'Amazon', 'Tesla'],
    'ticker': ['aapl', 'msft', None, 'goog', '', 'amzn', 'tsla']
})

# Clean numeric data
cleaned_prices = clean_numeric_data(
    df['price'],
    default_value=0.0,
    remove_outliers=True,
    outlier_threshold=2.0
)

print("Original prices:", df['price'].tolist())
print("Cleaned prices:", cleaned_prices.tolist())

# Clean string data
cleaned_companies = clean_string_data(
    df['company'],
    default_value="Unknown",
    strip_whitespace=True,
    convert_to_upper=False
)

print("Original companies:", df['company'].tolist())
print("Cleaned companies:", cleaned_companies.tolist())
```

### Numeric Range Validation

```python
import pandas as pd
from src.validation_utils import validate_numeric_range

# Sample data with out-of-range values
df = pd.DataFrame({
    'price': [50, 75, 100, 125, 150, 175, 200, 1000],  # 1000 is out of typical range
    'volume': [1000000, 500000, -100000, 2000000, 3000000, 4000000, 5000000, 10000000]
})

# Validate and constrain price range
validated_prices = validate_numeric_range(
    df['price'],
    min_value=0.0,
    max_value=500.0,
    allow_nan=False
)

# Validate and constrain volume range (no negative volumes)
validated_volumes = validate_numeric_range(
    df['volume'],
    min_value=0.0,
    max_value=None,  # No upper bound
    allow_nan=False
)

print("Original prices:", df['price'].tolist())
print("Validated prices:", validated_prices.tolist())

print("Original volumes:", df['volume'].tolist())
print("Validated volumes:", validated_volumes.tolist())
```

### List Validation

```python
from src.validation_utils import validate_list_of_dicts

# Sample list of financial data
financial_data = [
    {'date': '2023-01-01', 'ticker': 'AAPL', 'price': 150.0, 'volume': 1000000},
    {'date': '2023-01-02', 'ticker': 'MSFT', 'price': 250.0, 'volume': 1500000},
    {'date': '2023-01-03', 'ticker': 'GOOGL', 'price': 100.0, 'volume': 2000000},
    {'date': '2023-01-04', 'ticker': 'TSLA', 'price': 200.0, 'volume': 3000000}
]

# Validate the list of dictionaries
is_valid = validate_list_of_dicts(
    financial_data,
    required_keys=['date', 'ticker', 'price', 'volume'],
    min_length=1,
    max_length=100
)

print(f"Financial data valid: {is_valid}")

# Test with invalid data
invalid_data = [
    {'date': '2023-01-01', 'ticker': 'AAPL'},  # Missing price and volume
    {'date': '2023-01-02', 'price': 250.0, 'volume': 1500000}  # Missing ticker
]

is_valid = validate_list_of_dicts(
    invalid_data,
    required_keys=['date', 'ticker', 'price', 'volume']
)

print(f"Invalid data detected: {not is_valid}")
```

### Forecast Data Sanitization

```python
import pandas as pd
from src.validation_utils import sanitize_forecast_data

# Sample forecast data with issues
forecast_df = pd.DataFrame({
    'date': pd.date_range('2023-02-01', periods=5),
    'forecast_price': [110.0, None, 115.0, 117.0, 120.0],  # Has None value
    'confidence': [0.8, 1.5, 0.7, -0.2, 0.6],  # Has out-of-range confidence values
    'trend': ['UP', 'INVALID', 'DOWN', 'NEUTRAL', 'UP']  # Has invalid trend
})

print("Original forecast data:")
print(forecast_df)

# Sanitize forecast data
cleaned_df = sanitize_forecast_data(forecast_df)

print("\nSanitized forecast data:")
print(cleaned_df)
```

### Custom Validation

```python
import pandas as pd
from src.validation_utils import apply_custom_validation

# Sample financial data
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'price': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
})

def validate_financial_data(df):
    """Custom validation function for financial data."""
    # Check that all prices are positive
    if df['price'].min() <= 0:
        return False
    
    # Check that volume is reasonable (not too low, not too high)
    if df['volume'].min() < 100000:  # Minimum 100k volume
        return False
    
    if df['volume'].max() > 100000000:  # Maximum 100M volume
        return False
    
    # Check that we have reasonable price range
    price_range = df['price'].max() - df['price'].min()
    if price_range < 0.01:  # Minimum 1 cent range
        return False
    
    return True

# Apply custom validation
is_valid = apply_custom_validation(
    df,
    validate_financial_data,
    error_message="Financial data validation failed - check price and volume ranges"
)

print(f"Custom validation passed: {is_valid}")
```

---

## Search Utilities Examples

### Basic Web Search

```python
from src.search_utils import SearchUtils

# Initialize search utilities
search_utils = SearchUtils()

# Perform basic search
results = search_utils.web_search("Apple Inc stock information")

# Process results
print(f"Found {len(results)} search results:")
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Title: {result.title}")
    print(f"URL: {result.href}")
    print(f"Content: {result.content[:100]}...")
    print(f"Relevance: {result.relevance_score}")
```

### Search with Custom Configuration

```python
from src.search_utils import SearchUtils, SearchConfig

# Create custom search configuration
config = SearchConfig(
    max_results=10,
    timeout=60,
    region="us-en",
    safesearch="strict",
    retry_count=3,
    enable_fallback=True
)

# Initialize with custom config
search_utils = SearchUtils(config)

# Perform search
results = search_utils.web_search("Tesla Q4 2023 earnings")

print(f"Found {len(results)} results with custom config")
```

### Ticker Extraction from Search

```python
from src.search_utils import SearchUtils

search_utils = SearchUtils()

# Search for company information
company_name = "Apple Inc"
results = search_utils.web_search(f"{company_name} stock ticker")

# Extract ticker from search results
ticker = search_utils.parse_ticker_from_search(results, company_name)

print(f"Company: {company_name}")
print(f"Extracted ticker: {ticker}")
```

### Cryptocurrency Search and Classification

```python
from src.search_utils import SearchUtils

search_utils = SearchUtils()

# Classify asset type
asset_inputs = ["Bitcoin", "AAPL", "Ethereum", "Tesla", "XYZ"]

for input_text in asset_inputs:
    asset_type = search_utils.classify_asset_type(input_text)
    print(f"'{input_text}' is classified as: {asset_type}")
    
    if asset_type == "crypto":
        # Search for crypto information
        results = search_utils.web_search(f"{input_text} cryptocurrency")
        crypto_ticker = search_utils.parse_crypto_ticker_from_search(results, input_text)
        print(f"  Extracted crypto ticker: {crypto_ticker}")
        
        # Get CoinGecko ID
        coin_id = search_utils.parse_coingecko_id_from_search(results, input_text)
        print(f"  CoinGecko ID: {coin_id}")
```

### Search Text Extraction and Cleaning

```python
from src.search_utils import SearchUtils

search_utils = SearchUtils()

# Perform search
results = search_utils.web_search("Microsoft stock analysis 2024")

# Extract combined search text
search_text = search_utils.extract_search_text(results)
print(f"Combined search text length: {len(search_text)} characters")
print(f"First 200 characters: {search_text[:200]}...")

# Clean the text
cleaned_text = search_utils.clean_text(search_text)
print(f"Cleaned text length: {len(cleaned_text)} characters")
```

### Formatted Search Queries

```python
from src.search_utils import SearchUtils

search_utils = SearchUtils()

# Test different query types
queries = {
    "general": "artificial intelligence stocks",
    "ticker": "AAPL",
    "crypto": "BTC",
    "company": "NVIDIA Corporation",
    "crypto_id": "Phala Network"
}

for query_type, base_query in queries.items():
    formatted_query = search_utils.format_search_query(base_query, query_type)
    print(f"{query_type.upper()}: '{base_query}' -> '{formatted_query}'")
```

### Search with Retry Logic

```python
from src.search_utils import SearchUtils

search_utils = SearchUtils()

# Search with retry for unreliable queries
results = search_utils.search_with_retry(
    "obscure cryptocurrency token information",
    query_type="crypto",
    max_retries=3
)

if results:
    print(f"Search successful with {len(results)} results")
    for result in results[:3]:  # Show first 3 results
        print(f"- {result.title}")
else:
    print("Search failed after retries")
```

### Search Statistics

```python
from src.search_utils import SearchUtils

search_utils = SearchUtils()

# Get search statistics
stats = search_utils.get_search_stats()

print("Search Configuration:")
print(f"  Max results: {stats['config']['max_results']}")
print(f"  Timeout: {stats['config']['timeout']} seconds")
print(f"  Region: {stats['config']['region']}")
print(f"  Safe search: {stats['config']['safesearch']}")

print(f"\nGemini API configured: {stats['gemini_configured']}")
print(f"Timestamp: {stats['timestamp']}")
```

---

## JSON Helpers Examples

### Safe JSON Serialization

```python
from src.json_helpers import _dumps
import pandas as pd
import numpy as np
from datetime import datetime

# Complex data with various types
data = {
    'timestamp': datetime.now(),
    'pandas_timestamp': pd.Timestamp('2023-01-01'),
    'numpy_int': np.int64(42),
    'numpy_float': np.float64(3.14159),
    'numpy_array': np.array([1, 2, 3, 4, 5]),
    'pandas_series': pd.Series([1, 2, 3]),
    'dataframe': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
    'nested_dict': {
        'date': datetime.now(),
        'values': np.array([10, 20, 30])
    }
}

# Safe serialization
json_string = _dumps(data)
print("JSON serialization successful")
print(f"Length: {len(json_string)} characters")
```

### Formatted Display

```python
from src.json_helpers import _format_json_for_display

# Large data structure for display
large_data = {
    'financial_metrics': {
        'revenue': list(range(1000)),  # Large array
        'expenses': [i * 0.8 for i in range(1000)],
        'profit': [i * 0.2 for i in range(1000)]
    },
    'company_info': {
        'name': 'Very Long Company Name Incorporated',
        'description': 'This is an extremely long description that will be truncated for display purposes to keep the output readable and concise.',
        'employees': list(range(500))  # Another large array
    },
    'metadata': {
        'created': '2023-01-01',
        'updated': '2023-12-31'
    }
}

# Format for display
formatted = _format_json_for_display(large_data)
print("Formatted for display:")
print(formatted)
```

### JSON Extraction from Markdown

```python
from src.json_helpers import _extract_json_text, _parse_json_strictish

# JSON embedded in markdown
markdown_content = """
# API Response

Here's the JSON data:

```json
{
  "status": "success",
  "data": {
    "users": [
      {"id": 1, "name": "Alice"},
      {"id": 2, "name": "Bob"}
    ],
    "count": 2
  },
  "timestamp": "2023-12-01T10:30:00Z"
}
```

This is the end of the response.
"""

# Extract JSON text
json_text = _extract_json_text(markdown_content)
print("Extracted JSON:")
print(json_text)

# Parse the JSON
parsed_data = _parse_json_strictish(json_text)
print(f"\nParsed data type: {type(parsed_data)}")
print(f"Status: {parsed_data['status']}")
print(f"User count: {parsed_data['data']['count']}")
```

### Handling Trailing Commas

```python
from src.json_helpers import _clean_trailing_commas, _parse_json_strictish

# JSON with trailing commas (invalid standard JSON)
json_with_commas = """
{
  "company": "Tech Corp",
  "employees": [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
  ],
  "departments": {
    "engineering": 50,
    "sales": 20,
    "marketing": 10,
  },
  "founded": 2020,
}
"""

print("Original JSON (with trailing commas):")
print(json_with_commas)

# Clean trailing commas
cleaned_json = _clean_trailing_commas(json_with_commas)
print("\nCleaned JSON (trailing commas removed):")
print(cleaned_json)

# Parse the cleaned JSON
data = _parse_json_strictish(cleaned_json)
print(f"\nSuccessfully parsed data:")
print(f"Company: {data['company']}")
print(f"Employees: {len(data['employees'])}")
print(f"Departments: {list(data['departments'].keys())}")
```

---

## Complete Workflow Examples

### Stock Analysis Pipeline

```python
import pandas as pd
import yfinance as yf
from src.config import load_config, setup_logging
from src.decorators import handle_errors, log_execution, validate_inputs
from src.chart_utils import create_price_traces, create_chart_layout
from src.date_utils import sort_by_date, get_date_range
from src.validation_utils import validate_dataframe, clean_numeric_data
from src.search_utils import SearchUtils
import plotly.graph_objects as go

# Set up configuration and logging
config = load_config()
setup_logging()

class StockAnalyzer:
    def __init__(self):
        self.search_utils = SearchUtils()
    
    @handle_errors(default_return={})
    @log_execution(include_args=True)
    @validate_inputs(ticker='non_empty_string', days='positive_number')
    def analyze_stock(self, ticker: str, days: int = 30):
        """Complete stock analysis pipeline."""
        
        # Step 1: Validate and get company info
        print(f"Analyzing {ticker} for {days} days...")
        
        # Step 2: Fetch stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")
        
        if hist.empty:
            return {"error": "No data available"}
        
        # Step 3: Validate and clean data
        df = hist.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        # Validate DataFrame structure
        is_valid = validate_dataframe(
            df,
            required_columns=['date', 'open', 'high', 'low', 'close', 'volume'],
            column_types={
                'date': 'datetime',
                'open': 'numeric',
                'high': 'numeric',
                'low': 'numeric',
                'close': 'numeric',
                'volume': 'numeric'
            }
        )
        
        if not is_valid:
            return {"error": "Invalid data structure"}
        
        # Step 4: Sort by date and get range
        df = sort_by_date(df, 'date')
        date_range = get_date_range(df, 'date')
        
        # Step 5: Clean numeric data
        df['volume'] = clean_numeric_data(df['volume'])
        
        # Step 6: Calculate basic metrics
        current_price = df['close'].iloc[-1]
        price_change = df['close'].pct_change().iloc[-1]
        avg_volume = df['volume'].mean()
        volatility = df['close'].std()
        
        # Step 7: Get company information
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'Unknown')
        
        # Step 8: Create analysis result
        analysis = {
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'current_price': current_price,
            'price_change_pct': price_change * 100,
            'avg_volume': avg_volume,
            'volatility': volatility,
            'date_range': {
                'start': date_range[0].strftime('%Y-%m-%d'),
                'end': date_range[1].strftime('%Y-%m-%d')
            },
            'data_points': len(df)
        }
        
        return analysis
    
    @handle_errors(default_return=None)
    def create_price_chart(self, ticker: str, days: int = 30):
        """Create price chart for stock."""
        
        # Fetch data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")
        
        if hist.empty:
            return None
        
        # Prepare data
        df = hist.reset_index()
        df.columns = [col.lower() for col in df.columns]
        df = sort_by_date(df, 'date')
        
        # Create price traces
        price_traces = create_price_traces(df)
        
        # Create chart layout
        layout = create_chart_layout(
            title=f"{ticker} Stock Price - Last {days} Days",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600
        )
        
        # Create figure
        fig = go.Figure(data=price_traces, layout=layout)
        return fig

# Usage example
if __name__ == "__main__":
    analyzer = StockAnalyzer()
    
    # Analyze Apple stock
    analysis = analyzer.analyze_stock("AAPL", 30)
    print("\nStock Analysis Results:")
    for key, value in analysis.items():
        if key != 'date_range':
            print(f"  {key}: {value}")
    
    # Create chart
    chart = analyzer.create_price_chart("AAPL", 30)
    if chart:
        chart.show()
```

### Cryptocurrency Analysis with Search

```python
import pandas as pd
from pycoingecko import CoinGeckoAPI
from src.config import load_config
from src.decorators import handle_errors, validate_inputs
from src.search_utils import SearchUtils
from src.validation_utils import validate_dataframe, sanitize_forecast_data
from src.chart_utils import create_forecast_traces, hex_to_rgba
import plotly.graph_objects as go

# Set up configuration
config = load_config()
cg = CoinGeckoAPI()

class CryptoAnalyzer:
    def __init__(self):
        self.search_utils = SearchUtils()
    
    @handle_errors(default_return={})
    @validate_inputs(crypto_name='non_empty_string', days='positive_number')
    def analyze_crypto(self, crypto_name: str, days: int = 30):
        """Complete cryptocurrency analysis with search integration."""
        
        print(f"Analyzing {crypto_name} for {days} days...")
        
        # Step 1: Classify asset and search for information
        asset_type = self.search_utils.classify_asset_type(crypto_name)
        print(f"Asset type: {asset_type}")
        
        if asset_type != "crypto":
            return {"error": f"'{crypto_name}' is not classified as cryptocurrency"}
        
        # Step 2: Search for crypto ticker
        search_results = self.search_utils.web_search(f"{crypto_name} cryptocurrency")
        crypto_ticker = self.search_utils.parse_crypto_ticker_from_search(
            search_results, crypto_name
        )
        print(f"Extracted ticker: {crypto_ticker}")
        
        # Step 3: Get CoinGecko ID
        coin_id = self.search_utils.parse_coingecko_id_from_search(
            search_results, crypto_name
        )
        print(f"CoinGecko ID: {coin_id}")
        
        if not coin_id:
            return {"error": "Could not determine CoinGecko ID"}
        
        # Step 4: Fetch cryptocurrency data
        try:
            # Get current price and market data
            price_data = cg.get_coin_market_chart_by_id(
                id=coin_id, vs_currency='usd', days=days
            )
            
            # Get coin details
            coin_details = cg.get_coin_by_id(id=coin_id)
            
        except Exception as e:
            return {"error": f"Failed to fetch crypto data: {str(e)}"}
        
        # Step 5: Process price data
        prices = price_data['prices']
        if not prices:
            return {"error": "No price data available"}
        
        # Convert to DataFrame
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Validate data structure
        is_valid = validate_dataframe(
            df,
            required_columns=['date', 'price'],
            column_types={'date': 'datetime', 'price': 'numeric'}
        )
        
        if not is_valid:
            return {"error": "Invalid price data structure"}
        
        # Step 6: Calculate metrics
        current_price = df['price'].iloc[-1]
        price_change = ((df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]) * 100
        volatility = df['price'].std()
        avg_price = df['price'].mean()
        
        # Step 7: Extract coin information
        coin_info = {
            'name': coin_details.get('name', crypto_name),
            'symbol': coin_details.get('symbol', '').upper(),
            'market_cap': coin_details.get('market_data', {}).get('market_cap', {}).get('usd'),
            'volume_24h': coin_details.get('market_data', {}).get('total_volume', {}).get('usd'),
            'price_change_24h': coin_details.get('market_data', {}).get('price_change_percentage_24h'),
            'price_change_7d': coin_details.get('market_data', {}).get('price_change_percentage_7d')
        }
        
        # Step 8: Create analysis result
        analysis = {
            'crypto_name': crypto_name,
            'ticker': crypto_ticker,
            'coin_id': coin_id,
            'current_price': current_price,
            'price_change_pct': price_change,
            'volatility': volatility,
            'avg_price': avg_price,
            'data_points': len(df),
            'coin_info': coin_info
        }
        
        return analysis
    
    @handle_errors(default_return=None)
    def create_forecast_chart(self, forecast_data: dict):
        """Create forecast chart for cryptocurrency."""
        
        # Prepare forecast data
        dates = pd.date_range('2023-02-01', periods=7)
        forecast_prices = [
            forecast_data['current_price'] * (1 + 0.01 * i) 
            for i in range(7)
        ]
        confidence = [0.8 - 0.05 * i for i in range(7)]
        
        forecast_df = pd.DataFrame({
            'date': dates,
            'forecast_price': forecast_prices,
            'confidence': confidence
        })
        
        # Sanitize forecast data
        forecast_df = sanitize_forecast_data(forecast_df)
        
        # Create forecast traces
        forecast_traces = create_forecast_traces(forecast_df, line_color='#f39c12')
        
        # Create chart
        fig = go.Figure(data=forecast_traces)
        fig.update_layout(
            title=f"{forecast_data['crypto_name']} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600
        )
        
        return fig

# Usage example
if __name__ == "__main__":
    analyzer = CryptoAnalyzer()
    
    # Analyze Bitcoin
    analysis = analyzer.analyze_crypto("Bitcoin", 30)
    print("\nCryptocurrency Analysis Results:")
    for key, value in analysis.items():
        if key != 'coin_info':
            print(f"  {key}: {value}")
    
    # Create forecast chart
    if 'error' not in analysis:
        chart = analyzer.create_forecast_chart(analysis)
        if chart:
            chart.show()
```

---

## Migration Examples

### Before Migration (Old Code)

```python
# Old approach - all functions in services module
from src.services import (
    validate_ticker, get_company_info, load_prices,
    compute_indicators, build_report
)

# Manual error handling
try:
    ticker = validate_ticker("Apple Inc")
    company_info = get_company_info(ticker)
    prices = load_prices(ticker, 30)
    indicators = compute_indicators(prices)
    report = build_report(ticker, indicators, company_info)
except Exception as e:
    print(f"Error: {e}")
    report = None
```

### After Migration (New Code)

```python
# New approach - modular utilities with decorators
from src.decorators import handle_errors
from src.validation_utils import validate_dataframe
from src.search_utils import SearchUtils
import yfinance as yf

class ModernStockAnalyzer:
    def __init__(self):
        self.search_utils = SearchUtils()
    
    @handle_errors(default_return=None)
    def analyze_stock(self, input_text: str, days: int = 30):
        """Modern stock analysis with integrated utilities."""
        
        # Step 1: Search and validate ticker
        search_results = self.search_utils.web_search(f"{input_text} stock")
        ticker = self.search_utils.parse_ticker_from_search(
            search_results, input_text
        )
        
        if not ticker:
            return None
        
        # Step 2: Fetch data using modern libraries
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")
        
        # Step 3: Validate data structure
        df = hist.reset_index()
        is_valid = validate_dataframe(
            df,
            required_columns=['Date', 'Close'],
            column_types={'Date': 'datetime', 'Close': 'numeric'}
        )
        
        if not is_valid:
            return None
        
        # Step 4: Return analysis
        return {
            'ticker': ticker,
            'current_price': hist['Close'].iloc[-1],
            'price_change': hist['Close'].pct_change().iloc[-1],
            'volatility': hist['Close'].std()
        }

# Usage
analyzer = ModernStockAnalyzer()
result = analyzer.analyze_stock("Apple Inc", 30)
```

### Gradual Migration Strategy

```python
# Step 1: Enable compatibility warnings
from src.compatibility_wrappers import enable_compatibility_warnings
enable_compatibility_warnings()

# Step 2: Use both old and new approaches
from src.services import validate_ticker  # Old way (will show warning)
from src.search_utils import SearchUtils  # New way

search_utils = SearchUtils()

# Old approach (shows deprecation warning)
ticker_old = validate_ticker("Microsoft Corporation")

# New approach (recommended)
results = search_utils.web_search("Microsoft Corporation stock")
ticker_new = search_utils.parse_ticker_from_search(results, "Microsoft Corporation")

# Step 3: Gradually replace old calls with new ones
# ... continue migration ...

# Step 4: Disable compatibility layer when migration complete
from src.compatibility_wrappers import disable_compatibility_layer
disable_compatibility_layer()
```

### Configuration Migration

```python
# Old way - scattered environment variables
import os
os.environ['GEMINI_API_KEY'] = 'old-api-key'
os.environ['DEFAULT_DAYS'] = '30'

# New way - centralized configuration
from src.config import load_config

config = load_config()
config.gemini.api_key = 'new-api-key'
config.analysis.default_days = 60
config.feature_flags.enable_web_search = True

# Save for future use
config.save_to_file('updated_config.json')
```

These examples demonstrate the comprehensive functionality of the refactored utility modules and provide practical guidance for migration from the old codebase to the new modular architecture.