# Quick Start Guide: Refactored Utility Modules

This guide gets you up and running quickly with the new refactored utility modules in Agentic Ticker.

## Prerequisites

- Python 3.11+
- Google Gemini API key
- Basic familiarity with Python and financial data analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd agentic-ticker

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies for full functionality
pip install PyYAML  # For YAML configuration support
```

## Basic Setup

### 1. Set Up Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your Gemini API key
nano .env  # or use your preferred editor
```

**Required minimum environment variables:**
```bash
export GEMINI_API_KEY="your-actual-api-key"
export LOG_LEVEL="INFO"
```

### 2. Test Basic Configuration

```python
# test_config.py
from src.config import load_config, setup_logging

# Load configuration
config = load_config()
print(f"Configuration loaded: {config.gemini.model}")

# Set up logging
setup_logging()
print("Logging configured successfully")
```

## Quick Examples

### Example 1: Basic Stock Analysis

```python
# quick_stock_analysis.py
import yfinance as yf
from src.config import load_config, setup_logging
from src.decorators import handle_errors
from src.validation_utils import validate_dataframe
from src.chart_utils import create_price_traces, create_chart_layout
from src.date_utils import sort_by_date
import plotly.graph_objects as go

# Set up configuration and logging
config = load_config()
setup_logging()

@handle_errors(default_return=None)
def quick_stock_analysis(ticker: str, days: int = 30):
    """Quick stock analysis example."""
    
    # Fetch data
    stock = yf.Ticker(ticker)
    hist = stock.history(period=f"{days}d")
    
    if hist.empty:
        return None
    
    # Prepare data
    df = hist.reset_index()
    df.columns = [col.lower() for col in df.columns]
    df = sort_by_date(df, 'date')
    
    # Validate data
    is_valid = validate_dataframe(df, required_columns=['date', 'close'])
    if not is_valid:
        return None
    
    # Create analysis
    current_price = df['close'].iloc[-1]
    price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
    
    # Create chart
    traces = create_price_traces(df)
    layout = create_chart_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)"
    )
    
    fig = go.Figure(data=traces, layout=layout)
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'price_change_pct': price_change,
        'chart': fig
    }

# Run analysis
if __name__ == "__main__":
    result = quick_stock_analysis("AAPL", 30)
    if result:
        print(f"{result['ticker']}: ${result['current_price']:.2f}")
        print(f"Price change: {result['price_change_pct']:.1f}%")
        result['chart'].show()
```

### Example 2: Web Search and Ticker Validation

```python
# quick_search.py
from src.search_utils import SearchUtils
from src.config import load_config, setup_logging

# Set up configuration
config = load_config()
setup_logging()

# Initialize search utilities
search_utils = SearchUtils()

def quick_search_example():
    """Quick web search example."""
    
    # Search for company information
    company_name = "Apple Inc"
    results = search_utils.web_search(f"{company_name} stock ticker")
    
    print(f"Found {len(results)} search results for '{company_name}':")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result.title}")
        print(f"   {result.href}")
        print(f"   {result.content[:100]}...")
    
    # Extract ticker from search results
    ticker = search_utils.parse_ticker_from_search(results, company_name)
    print(f"\nExtracted ticker: {ticker}")
    
    # Classify asset type
    asset_type = search_utils.classify_asset_type(company_name)
    print(f"Asset type: {asset_type}")

if __name__ == "__main__":
    quick_search_example()
```

### Example 3: Data Validation and Cleaning

```python
# quick_validation.py
import pandas as pd
import numpy as np
from src.validation_utils import validate_dataframe, clean_numeric_data
from src.config import load_config, setup_logging

# Set up configuration
config = load_config()
setup_logging()

def quick_validation_example():
    """Quick data validation example."""
    
    # Create sample data with issues
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'price': [100, 101, None, 103, np.nan, 105, 106, 107, 108, 109],
        'volume': [1000000, 1100000, -100000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
    })
    
    print("Original data:")
    print(df.head())
    
    # Validate DataFrame structure
    is_valid = validate_dataframe(
        df,
        required_columns=['date', 'price', 'volume'],
        column_types={'date': 'datetime', 'price': 'numeric', 'volume': 'numeric'}
    )
    print(f"DataFrame valid: {is_valid}")
    
    # Clean numeric data
    df['price'] = clean_numeric_data(df['price'], default_value=0.0)
    df['volume'] = clean_numeric_data(df['volume'], min_value=0)
    
    print("\nCleaned data:")
    print(df.head())

if __name__ == "__main__":
    quick_validation_example()
```

### Example 4: Decorator Usage

```python
# quick_decorators.py
from src.decorators import (
    handle_errors, log_execution, time_execution, 
    validate_inputs, cache_result
)
from src.config import load_config, setup_logging
import time

# Set up configuration
config = load_config()
setup_logging()

@handle_errors(default_return="Error occurred")
@log_execution(include_args=True)
@time_execution(log_threshold=0.1)
@validate_inputs(name='non_empty_string', count='positive_number')
@cache_result(max_size=8)
def quick_decorated_function(name: str, count: int):
    """Example function with multiple decorators."""
    
    # Simulate some work
    time.sleep(0.2)
    
    result = f"Hello {name}! Count: {count}"
    return result

def quick_decorator_example():
    """Quick decorator example."""
    
    # First call (will be cached)
    result1 = quick_decorated_function("Alice", 5)
    print(f"Result 1: {result1}")
    
    # Second call (will use cache)
    result2 = quick_decorated_function("Alice", 5)
    print(f"Result 2: {result2}")
    
    # Different parameters (new execution)
    result3 = quick_decorated_function("Bob", 10)
    print(f"Result 3: {result3}")

if __name__ == "__main__":
    quick_decorator_example()
```

## Common Use Cases

### Use Case 1: Financial Data Pipeline

```python
# financial_pipeline.py
import pandas as pd
import yfinance as yf
from src.config import load_config
from src.decorators import handle_errors
from src.validation_utils import validate_dataframe, clean_numeric_data
from src.date_utils import sort_by_date, get_date_range
from src.chart_utils import create_price_traces, create_chart_layout
import plotly.graph_objects as go

@handle_errors(default_return=None)
def financial_data_pipeline(ticker: str, period: str = "1mo"):
    """Complete financial data processing pipeline."""
    
    # Load configuration
    config = load_config()
    
    # Fetch data
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        return None
    
    # Process data
    df = hist.reset_index()
    df.columns = [col.lower() for col in df.columns]
    df = sort_by_date(df, 'date')
    
    # Validate data
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
        return None
    
    # Clean data
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = clean_numeric_data(df[col])
    
    # Get date range
    date_range = get_date_range(df, 'date')
    
    # Create visualization
    traces = create_price_traces(df)
    layout = create_chart_layout(
        title=f"{ticker} Financial Data",
        xaxis_title="Date",
        yaxis_title="Price ($)"
    )
    
    fig = go.Figure(data=traces, layout=layout)
    
    return {
        'data': df,
        'date_range': date_range,
        'chart': fig,
        'summary': {
            'rows': len(df),
            'start_date': date_range[0] if date_range else None,
            'end_date': date_range[1] if date_range else None
        }
    }

if __name__ == "__main__":
    result = financial_data_pipeline("MSFT", "3mo")
    if result:
        print(f"Processed {result['summary']['rows']} rows")
        print(f"Date range: {result['summary']['start_date']} to {result['summary']['end_date']}")
        result['chart'].show()
```

### Use Case 2: Search and Analysis Integration

```python
# search_analysis.py
from src.search_utils import SearchUtils
from src.config import load_config
from src.decorators import handle_errors
import yfinance as yf

@handle_errors(default_return=None)
def search_and_analyze(asset_name: str):
    """Search for asset information and perform analysis."""
    
    # Initialize search utilities
    search_utils = SearchUtils()
    config = load_config()
    
    # Search for asset information
    search_results = search_utils.web_search(f"{asset_name} financial data")
    
    # Classify asset type
    asset_type = search_utils.classify_asset_type(asset_name)
    print(f"Asset '{asset_name}' classified as: {asset_type}")
    
    # Extract relevant information based on asset type
    if asset_type == "stock":
        ticker = search_utils.parse_ticker_from_search(search_results, asset_name)
        print(f"Extracted ticker: {ticker}")
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period=f"{config.analysis.default_days}d")
        
        return {
            'type': 'stock',
            'ticker': ticker,
            'name': info.get('longName', asset_name),
            'sector': info.get('sector', 'Unknown'),
            'current_price': hist['Close'].iloc[-1] if not hist.empty else None,
            'search_results': len(search_results)
        }
    
    elif asset_type == "crypto":
        crypto_ticker = search_utils.parse_crypto_ticker_from_search(search_results, asset_name)
        coin_id = search_utils.parse_coingecko_id_from_search(search_results, asset_name)
        
        print(f"Extracted crypto ticker: {crypto_ticker}")
        print(f"Extracted CoinGecko ID: {coin_id}")
        
        return {
            'type': 'crypto',
            'ticker': crypto_ticker,
            'coin_id': coin_id,
            'name': asset_name,
            'search_results': len(search_results)
        }
    
    else:
        return {
            'type': 'ambiguous',
            'name': asset_name,
            'search_results': len(search_results),
            'suggestion': 'Try providing a ticker symbol (e.g., AAPL, BTC)'
        }

if __name__ == "__main__":
    # Test with different asset types
    assets = ["Apple Inc", "Bitcoin", "Tesla", "Ethereum"]
    
    for asset in assets:
        print(f"\nAnalyzing: {asset}")
        result = search_and_analyze(asset)
        if result:
            for key, value in result.items():
                print(f"  {key}: {value}")
```

## Next Steps

### 1. Explore the Documentation

- **[API Documentation](API_DOCUMENTATION.md)**: Complete API reference
- **[Usage Examples](USAGE_EXAMPLES.md)**: Detailed examples and workflows
- **[Configuration Guide](CONFIGURATION_GUIDE.md)**: Advanced configuration options
- **[Performance Benchmarks](PERFORMANCE_BENCHMARKS.md)**: Performance metrics and improvements
- **[Troubleshooting](TROUBLESHOOTING.md)**: Common issues and solutions

### 2. Run the Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_data_models.py
pytest tests/test_integration.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 3. Try the Full Application

```bash
# Run the main Streamlit application
streamlit run agentic_ticker.py

# Or run with custom configuration
CONFIG_FILE=my_config.json streamlit run agentic_ticker.py
```

### 4. Customize for Your Needs

```python
# Create custom configuration
from src.config import load_config, AppConfig

config = load_config()
config.gemini.model = "gemini-pro"  # Use different model
config.analysis.default_days = 90   # Longer analysis period
config.feature_flags.enable_animations = False  # Disable for performance
config.save_to_file("my_config.json")
```

## Getting Help

If you encounter issues:

1. **Check the troubleshooting guide**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **Verify your setup**: Run the test examples above
3. **Check logs**: Enable debug logging with `LOG_LEVEL=DEBUG`
4. **Review configuration**: Ensure all required settings are configured

For additional support:
- Review the comprehensive documentation in the `docs/` directory
- Check the examples in this quick start guide
- Examine the test files for working code examples

---

**Happy analyzing! 🚀**

This quick start guide should get you productive with the refactored utility modules quickly. For more detailed information, refer to the comprehensive documentation files.