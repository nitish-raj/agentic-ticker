import pytest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Note: Tests should use environment variables or mock the config system.


@pytest.fixture
def sample_price_data():
    """Sample price data for testing"""
    return [
        {
            "date": "2023-01-01",
            "close": 100,
            "open": 99,
            "high": 101,
            "low": 98,
            "volume": 1000000,
        },
        {
            "date": "2023-01-02",
            "close": 101,
            "open": 100,
            "high": 102,
            "low": 99,
            "volume": 1100000,
        },
        {
            "date": "2023-01-03",
            "close": 102,
            "open": 101,
            "high": 103,
            "low": 100,
            "volume": 1200000,
        },
        {
            "date": "2023-01-04",
            "close": 103,
            "open": 102,
            "high": 104,
            "low": 101,
            "volume": 1300000,
        },
        {
            "date": "2023-01-05",
            "close": 104,
            "open": 103,
            "high": 105,
            "low": 102,
            "volume": 1400000,
        },
    ]


@pytest.fixture
def sample_indicator_data():
    """Sample indicator data for testing"""
    return [
        {
            "date": "2023-01-03",
            "ma5": 101.0,
            "ma10": 100.5,
            "daily_return": 0.01,
            "volatility": 0.02,
        },
        {
            "date": "2023-01-04",
            "ma5": 102.0,
            "ma10": 101.0,
            "daily_return": 0.02,
            "volatility": 0.02,
        },
        {
            "date": "2023-01-05",
            "ma5": 103.0,
            "ma10": 101.5,
            "daily_return": 0.01,
            "volatility": 0.02,
        },
    ]


@pytest.fixture
def sample_events():
    """Sample events for testing"""
    return [
        {
            "date": "2023-01-04",
            "price": 102.0,
            "change_percent": 5.0,
            "direction": "UP",
        },
        {
            "date": "2023-01-05",
            "price": 103.0,
            "change_percent": -3.0,
            "direction": "DOWN",
        },
    ]


@pytest.fixture
def sample_forecasts():
    """Sample forecasts for testing"""
    return [
        {
            "date": "2023-01-06",
            "forecast_price": 104.0,
            "confidence": 0.8,
            "trend": "UP",
        },
        {
            "date": "2023-01-07",
            "forecast_price": 105.0,
            "confidence": 0.7,
            "trend": "UP",
        },
    ]


@pytest.fixture
def sample_company_info():
    """Sample company info for testing"""
    return {
        "company_name": "Apple Inc.",
        "short_name": "Apple",
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }


@pytest.fixture
def sample_crypto_info():
    """Sample crypto info for testing"""
    return {
        "name": "Bitcoin",
        "symbol": "btc",
        "current_price_usd": 50000.0,
        "market_cap_usd": 1000000000000.0,
        "price_change_percentage_24h": 2.5,
        "price_change_percentage_7d": 5.0,
        "total_volume_usd": 30000000000.0,
    }
