"""
Compatibility Wrappers for Existing Functions

This module provides wrapper functions that maintain existing function signatures
while providing backward compatibility and migration path to new utility modules.
"""

import warnings
import functools
from typing import Any, Dict, List, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)


# Simple compatibility configuration
class SimpleCompatibilityConfig:
    """Simple compatibility configuration"""
    def __init__(self):
        self.enabled = True
        self.show_warnings = True
        self.strict_mode = False

compatibility_config = SimpleCompatibilityConfig()


def simple_deprecation_warning(message: str, stacklevel: int = 2) -> None:
    """Simple deprecation warning"""
    if not compatibility_config.show_warnings:
        return

    if compatibility_config.strict_mode:
        raise DeprecationWarning(message)

    warnings.warn(
        f"[DEPRECATED] {message}",
        DeprecationWarning,
        stacklevel=stacklevel
    )


def simple_compatibility_wrapper(
    func, new_func_name: Optional[str] = None, migration_guide: Optional[str] = None
):
    """Simple compatibility wrapper decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not compatibility_config.enabled:
            return func(*args, **kwargs)

        # Issue deprecation warning
        func_name = func.__name__
        module_name = func.__module__

        warning_msg = f"{module_name}.{func_name} is deprecated"
        if new_func_name:
            warning_msg += f", use {new_func_name} instead"

        if migration_guide:
            warning_msg += f". See: {migration_guide}"

        simple_deprecation_warning(warning_msg, stacklevel=3)

        # Call original function
        return func(*args, **kwargs)

    return wrapper


# Migration guide URL
MIGRATION_GUIDE = "https://github.com/your-org/agentic-ticker/blob/main/docs/MIGRATION_GUIDE.md"


# Import original functions with error handling
def _import_original_functions():
    """Import original functions with error handling"""
    try:
        import importlib
        services_module = importlib.import_module('src.services')
        return {
            'validate_ticker': getattr(services_module, 'validate_ticker', None),
            'get_company_info': getattr(services_module, 'get_company_info', None),
            'get_crypto_info': getattr(services_module, 'get_crypto_info', None),
            'load_prices': getattr(services_module, 'load_prices', None),
            'load_crypto_prices': getattr(services_module, 'load_crypto_prices', None),
            'compute_indicators': getattr(services_module, 'compute_indicators', None),
            'detect_events': getattr(services_module, 'detect_events', None),
            'forecast_prices': getattr(services_module, 'forecast_prices', None),
            'build_report': getattr(services_module, 'build_report', None),
            'ddgs_search': getattr(services_module, 'ddgs_search', None),
            'classify_asset_type': getattr(services_module, 'classify_asset_type', None),
            'generate_analysis_summary': getattr(services_module, 'generate_analysis_summary', None),
            'convert_yahoo_ticker_to_coingecko_id': getattr(
                services_module, 'convert_yahoo_ticker_to_coingecko_id', None
            ),
        }
    except ImportError as e:
        logger.warning(f"Could not import original functions: {e}")
        return {}


# Get original functions
_original_functions = _import_original_functions()


# Wrapper functions with compatibility layer

@simple_compatibility_wrapper(
    lambda *args, **kwargs: "",
    new_func_name=(
        "utility_modules.validation.validate_ticker"
    ),
    migration_guide=MIGRATION_GUIDE
)
def validate_ticker(input_text: str) -> str:
    """
Validates and converts stock or crypto name/ticker to proper ticker symbol.

[DEPRECATED] This function is deprecated. Use utility_modules.validation.validate_ticker instead.

    Args:
        input_text: User input (can be ticker symbol like 'AAPL'/'BTC' or name like
        'Apple'/'Bitcoin')
    Returns:
        Valid ticker symbol (e.g., 'AAPL' for stocks, 'BTC-USD' for crypto)
    """
    original_func = _original_functions.get('validate_ticker')
    if original_func:
        return original_func(input_text)
    return ""


@simple_compatibility_wrapper(
    _original_functions.get(
        'get_company_info', lambda *args, **kwargs: {}
    ),
    new_func_name=(
        "utility_modules.company_info.get_company_info"
    ),
    migration_guide=MIGRATION_GUIDE
)
def get_company_info(ticker: str) -> Dict[str, str]:
    """
    Get company name and basic info for a ticker.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.company_info.get_company_info instead.
    
    Args:
        ticker: Stock ticker symbol
    Returns:
        Dict with company info
    """
    original_func = _original_functions.get('get_company_info')
    if original_func:
        return original_func(ticker)
    return {}


@simple_compatibility_wrapper(
    _original_functions.get(
        'get_crypto_info', lambda *args, **kwargs: {}
    ),
    new_func_name=(
        "utility_modules.crypto_info.get_crypto_info"
    ),
    migration_guide=MIGRATION_GUIDE
)
def get_crypto_info(ticker: str, original_input: str = "") -> Dict[str, Any]:
    """
    Get basic information about a cryptocurrency using CoinGecko API.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.crypto_info.get_crypto_info instead.
    
    Args:
        ticker: Cryptocurrency ticker symbol (e.g., 'bitcoin', 'ethereum', 'btc')
        original_input: Original user input for better resolution
    Returns:
        Dict with cryptocurrency info including name, symbol, market cap, etc.
    """
    original_func = _original_functions.get('get_crypto_info')
    if original_func:
        return original_func(ticker, original_input)
    return {}

@simple_compatibility_wrapper(
    _original_functions.get('load_prices', lambda *args, **kwargs: []),
    new_func_name="utility_modules.data_loading.load_prices",
    migration_guide=MIGRATION_GUIDE
)
def load_prices(ticker: str, days: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches historical OHLC data for a ticker over N days.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.data_loading.load_prices instead.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data to fetch
    Returns:
        List of dicts: {ticker,date,open,high,low,close,volume}
    """
    original_func = _original_functions.get('load_prices')
    if original_func:
        return original_func(ticker, days)
    return []

@simple_compatibility_wrapper(
    _original_functions.get('load_crypto_prices', lambda *args, **kwargs: []),
    new_func_name="utility_modules.data_loading.load_crypto_prices",
    migration_guide=MIGRATION_GUIDE
)
def load_crypto_prices(ticker: str, days: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches historical OHLC data for a cryptocurrency ticker over N days.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.data_loading.load_crypto_prices instead.
    
    Args:
        ticker: Crypto ticker symbol (e.g., 'BTC-USD')
        days: Number of days of historical data to fetch
    Returns:
        List of dicts: {ticker,date,open,high,low,close,volume}
    """
    original_func = _original_functions.get('load_crypto_prices')
    if original_func:
        return original_func(ticker, days)
    return []

@simple_compatibility_wrapper(
    _original_functions.get('compute_indicators', lambda *args, **kwargs: []),
    new_func_name="utility_modules.analysis.compute_indicators",
    migration_guide=MIGRATION_GUIDE
)
def compute_indicators(price_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculates moving averages, daily returns, and volatility based on available data.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.analysis.compute_indicators instead.
    
    Args:
        price_data: Output list from load_prices
    Returns:
        List of dicts: {date,ma5,ma10,daily_return,volatility}
    """
    return _original_functions['compute_indicators'](price_data)

@simple_compatibility_wrapper(
    _original_functions.get('detect_events', lambda *args, **kwargs: []),
    new_func_name="utility_modules.analysis.detect_events",
    migration_guide=MIGRATION_GUIDE
)
def detect_events(indicator_data: List[Dict[str, Any]], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Flags price movements where |Δ| >= threshold%.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.analysis.detect_events instead.
    
    Args:
        indicator_data: Output from compute_indicators
        threshold: Percentage threshold for event detection
    Returns:
        List of dicts: {date,price,change_percent,direction}
    """
    return _original_functions['detect_events'](indicator_data, threshold)

@simple_compatibility_wrapper(
    _original_functions.get('forecast_prices', lambda *args, **kwargs: []),
    new_func_name="utility_modules.analysis.forecast_prices",
    migration_guide=MIGRATION_GUIDE
)
def forecast_prices(indicator_data: List[Dict[str, Any]], days: int = 5) -> List[Dict[str, Any]]:
    """
    Simple price forecasting based on recent trends and indicators.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.analysis.forecast_prices instead.
    
    Args:
        indicator_data: Output from compute_indicators
        days: Number of days to forecast
    Returns:
        List of dicts: {date,forecast_price,confidence}
    """
    return _original_functions['forecast_prices'](indicator_data, days)

@simple_compatibility_wrapper(
    _original_functions.get('build_report', lambda *args, **kwargs: {}),
    new_func_name="utility_modules.reporting.build_report",
    migration_guide=MIGRATION_GUIDE
)
def build_report(ticker: str, events: List[Dict[str, Any]], forecasts: List[Dict[str, Any]], 
                company_info: Optional[Dict[str, str]] = None, 
                crypto_info: Optional[Dict[str, Any]] = None,
                price_data: Optional[List[Dict[str, Any]]] = None,
                indicator_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Generates a markdown brief with events and price forecasts.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.reporting.build_report instead.
    
    Args:
        ticker: Stock ticker symbol
        events: Output from detect_events
        forecasts: Output from forecast_prices
        company_info: Optional company information
        crypto_info: Optional cryptocurrency information
    Returns:
        Dict with keys {ticker,analysis_period,generated_date,content}
    """
    return _original_functions['build_report'](ticker, events, forecasts, company_info, crypto_info, price_data, indicator_data)

@simple_compatibility_wrapper(
    _original_functions.get('ddgs_search', lambda *args, **kwargs: []),
    new_func_name="utility_modules.search.ddgs_search",
    migration_guide=MIGRATION_GUIDE
)
def ddgs_search(query, max_results=3, **kwargs):
    """
    Search using DDGS (DuckDuckGo Search) library.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.search.ddgs_search instead.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        **kwargs: Additional search parameters
    Returns:
        List of search results
    """
    return _original_functions['ddgs_search'](query, max_results, **kwargs)

@simple_compatibility_wrapper(
    _original_functions.get('classify_asset_type', lambda *args, **kwargs: "ambiguous"),
    new_func_name="utility_modules.classification.classify_asset_type",
    migration_guide=MIGRATION_GUIDE
)
def classify_asset_type(input_text: str) -> str:
    """
    Classify asset type using Gemini API to determine if input refers to a stock, cryptocurrency, or is ambiguous.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.classification.classify_asset_type instead.
    
    Args:
        input_text: User input text to classify
    Returns:
        'stock', 'crypto', or 'ambiguous'
    """
    return _original_functions['classify_asset_type'](input_text)

@simple_compatibility_wrapper(
    _original_functions.get('generate_analysis_summary', lambda *args, **kwargs: ""),
    new_func_name="utility_modules.reporting.generate_analysis_summary",
    migration_guide=MIGRATION_GUIDE
)
def generate_analysis_summary(ticker: str, events: List[Dict[str, Any]], forecasts: List[Dict[str, Any]]) -> str:
    """
    Generate a dynamic analysis summary using Gemini.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.reporting.generate_analysis_summary instead.
    
    Args:
        ticker: Stock ticker symbol
        events: List of detected events
        forecasts: List of price forecasts
    Returns:
        Analysis summary string
    """
    return _original_functions['generate_analysis_summary'](ticker, events, forecasts)

@simple_compatibility_wrapper(
    _original_functions.get('convert_yahoo_ticker_to_coingecko_id', lambda *args, **kwargs: ""),
    new_func_name="utility_modules.crypto_utils.convert_yahoo_ticker_to_coingecko_id",
    migration_guide=MIGRATION_GUIDE
)
def convert_yahoo_ticker_to_coingecko_id(yahoo_ticker: str, original_input: str = "") -> str:
    """
    Convert Yahoo Finance crypto ticker format to CoinGecko coin ID using dynamic resolution.
    
    [DEPRECATED] This function is deprecated. Use utility_modules.crypto_utils.convert_yahoo_ticker_to_coingecko_id instead.
    
    Args:
        yahoo_ticker: Yahoo Finance ticker (e.g., 'DOGE-USD', 'BTC-USD')
        original_input: Original user input (e.g., 'PHALA NETWORK', 'DOGECOIN')
    Returns:
        CoinGecko coin ID (e.g., 'dogecoin', 'bitcoin', 'phala-network')
    """
    return _original_functions['convert_yahoo_ticker_to_coingecko_id'](yahoo_ticker, original_input)

# Compatibility module exports
__all__ = [
    'validate_ticker',
    'get_company_info', 
    'get_crypto_info',
    'load_prices',
    'load_crypto_prices',
    'compute_indicators',
    'detect_events',
    'forecast_prices',
    'build_report',
    'ddgs_search',
    'classify_asset_type',
    'generate_analysis_summary',
    'convert_yahoo_ticker_to_coingecko_id',
    'compatibility_config',
    'simple_deprecation_warning',
    'simple_compatibility_wrapper',
]

# Utility functions for compatibility management
def enable_compatibility_warnings() -> None:
    """Enable deprecation warnings"""
    compatibility_config.show_warnings = True

def disable_compatibility_warnings() -> None:
    """Disable deprecation warnings"""
    compatibility_config.show_warnings = False

def set_strict_mode(strict: bool = True) -> None:
    """Enable or disable strict mode"""
    compatibility_config.strict_mode = strict

def enable_compatibility_layer() -> None:
    """Enable compatibility layer"""
    compatibility_config.enabled = True

def disable_compatibility_layer() -> None:
    """Disable compatibility layer"""
    compatibility_config.enabled = False

def get_compatibility_status() -> Dict[str, Any]:
    """Get current compatibility status"""
    return {
        "enabled": compatibility_config.enabled,
        "show_warnings": compatibility_config.show_warnings,
        "strict_mode": compatibility_config.strict_mode,
        "available_functions": len([f for f in _original_functions.values() if f is not None]),
        "total_functions": len(_original_functions),
    }