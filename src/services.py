import os
import re
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

try:
    from coingecko_sdk import Coingecko
    CoinGeckoAPI = Coingecko
except ImportError:
    CoinGeckoAPI = None


def ddgs_search(query, max_results=3, **kwargs):
    """Search using DDGS (DuckDuckGo Search) library"""
    try:
        from ddgs import DDGS
        
        # Initialize DDGS with default settings
        ddgs = DDGS()
        
        # Perform text search
        results = ddgs.text(query, region='us-en', safesearch='moderate', max_results=max_results)
        
        # Convert DDGS results to expected format
        formatted_results = []
        for result in results:
            formatted_result = {
                'title': result.get('title', ''),
                'href': result.get('href', ''),
                'content': result.get('body', '')
            }
            formatted_results.append(formatted_result)
        
        print(f"✓ Web search returned {len(formatted_results)} results for query: {query}")
        return formatted_results[:max_results]
        
    except Exception as e:
        print(f"Web search failed: {e}")
        return []


def validate_crypto_ticker(input_text: str) -> str:
    """
    Validates and converts crypto name or ticker to proper ticker symbol using Gemini and CoinGecko.
    Args:
        input_text: User input (can be ticker symbol like 'BTC' or crypto name like 'Bitcoin')
    Returns:
        Valid crypto ticker symbol (e.g., 'BTC-USD' for Yahoo Finance format)
    """
    try:
        # Handle empty input
        if not input_text or not input_text.strip():
            return ""
        
        # Clean and normalize input
        cleaned_input = input_text.strip().upper()
        
        # Check if it's already a common crypto ticker format
        # Use Gemini to resolve the crypto ticker dynamically
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Fallback: return with -USD suffix for dynamic resolution
            return f"{cleaned_input}-USD"
            
        model = os.getenv("GEMINI_MODEL")
        api_base = os.getenv("GEMINI_API_BASE")
        
        prompt = f"""
        The user entered: "{input_text}"
        
        If this is already a valid cryptocurrency ticker symbol (like BTC, ETH, XRP, ADA), return it in uppercase.
        If this is a cryptocurrency name (like Bitcoin, Ethereum, Ripple, Cardano), return the correct ticker symbol.
        
        For cryptocurrency tickers, use the standard symbol (BTC, ETH, XRP, etc.) without any currency suffix.
        
        Return ONLY the ticker symbol in uppercase, nothing else. No explanations, no formatting.
        
        Examples:
        - "BTC" -> "BTC"
        - "Bitcoin" -> "BTC"
        - "ETH" -> "ETH" 
        - "Ethereum" -> "ETH"
        - "XRP" -> "XRP"
        - "Ripple" -> "XRP"
        - "ADA" -> "ADA"
        - "Cardano" -> "ADA"
        - "DOGE" -> "DOGE"
        - "Dogecoin" -> "DOGE"
        """
        
        url = f"{api_base}/models/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "text/plain"
            }
        }
        
        r = requests.post(url, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # Extract the ticker from Gemini response
        ticker = data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
        
        # Clean up ticker (remove spaces, special characters except alphanumeric)
        ticker = re.sub(r'[^A-Z0-9]', '', ticker)
        
        # Validate the returned ticker with Yahoo Finance format
        yf_ticker = f"{ticker}-USD"
        
        try:
            test_ticker = yf.Ticker(yf_ticker)
            test_data = test_ticker.history(period="1d")
            if not test_data.empty:
                return yf_ticker
        except Exception as e:
            print(f"Yahoo Finance validation failed for {yf_ticker}: {e}")
        
        # If Yahoo Finance fails, try CoinGecko validation
        try:
            if CoinGeckoAPI:
                # Get API key from environment
                demo_api_key = os.getenv("COINGECKO_DEMO_API_KEY")
                if demo_api_key:
                    cg = CoinGeckoAPI(demo_api_key=demo_api_key)
                else:
                    # Try pro API key if available
                    api_key = os.getenv("COINGECKO_API_KEY")
                    if api_key:
                        cg = CoinGeckoAPI(pro_api_key=api_key)
                    else:
                        cg = CoinGeckoAPI()
                
                # Try to get coin data
                coin_data = cg.coins.get_id(id=ticker.lower())
                if coin_data:
                    # CoinGecko found it, so it's valid
                    return yf_ticker
        except Exception as e:
            print(f"CoinGecko validation failed for {ticker}: {e}")
        
        # If Gemini fails or returns invalid ticker, try web search as fallback
        try:
            print(f"Attempting web search for crypto: {input_text}")
            search_results = ddgs_search(
                query=f"{input_text} cryptocurrency ticker symbol",
                max_results=5
            )
            
            if search_results and len(search_results) > 0:
                # Extract ticker from search results
                search_text = " ".join([result.get('title', '') + ' ' + result.get('content', '') for result in search_results])
                
                # Use Gemini to parse the search results and extract the ticker
                parse_prompt = f"""
                Based on these search results about "{input_text}", extract the correct cryptocurrency ticker symbol.
                
                Search results:
                {search_text}
                
                Return ONLY the ticker symbol in uppercase, nothing else. No explanations.
                Examples: "BTC", "ETH", "XRP", "ADA"
                """
                
                parse_body = {
                    "contents": [{"role": "user", "parts": [{"text": parse_prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "responseMimeType": "text/plain"
                    }
                }
                
                parse_r = requests.post(url, json=parse_body, timeout=30)
                parse_r.raise_for_status()
                parse_data = parse_r.json()
                
                parsed_ticker = parse_data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
                parsed_ticker = re.sub(r'[^A-Z0-9]', '', parsed_ticker)
                
                # Validate the parsed ticker
                yf_parsed_ticker = f"{parsed_ticker}-USD"
                try:
                    test_ticker = yf.Ticker(yf_parsed_ticker)
                    test_data = test_ticker.history(period="1d")
                    if not test_data.empty:
                        print(f"✓ Web search + Gemini parsing successful: {yf_parsed_ticker}")
                        return yf_parsed_ticker
                except Exception as e:
                    print(f"Parsed crypto ticker validation failed for {yf_parsed_ticker}: {e}")
            else:
                print("⚠ No search results found for crypto")
                        
        except Exception as e:
            print(f"Crypto web search fallback failed: {e}")
        
        # If all else fails, return empty string
        return ""
        
    except Exception as e:
        print(f"Crypto validation failed: {e}")
        return ""


def validate_ticker(input_text: str) -> str:
    """
    Validates and converts stock or crypto name/ticker to proper ticker symbol.
    Args:
        input_text: User input (can be ticker symbol like 'AAPL'/'BTC' or name like 'Apple'/'Bitcoin')
    Returns:
        Valid ticker symbol (e.g., 'AAPL' for stocks, 'BTC-USD' for crypto)
    """
    try:
        # Handle empty input
        if not input_text or not input_text.strip():
            return ""
        
        # Classify the asset type first
        asset_type = classify_asset_type(input_text)
        
        # Route to appropriate validation function based on asset type
        if asset_type == "crypto":
            return validate_crypto_ticker(input_text)
        else:
            # Handle stock validation (existing logic)
            # Check if it's already a valid ticker format (1-5 characters, letters only)
            if re.match(r'^[A-Z]{1,5}$', input_text.upper()):
                # Quick validation - try to fetch data
                test_ticker = yf.Ticker(input_text.upper())
                test_data = test_ticker.history(period="1d")
                if not test_data.empty:
                    return input_text.upper()
        
        # Use Gemini to resolve the ticker
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # No fallback - require Gemini API for dynamic validation
            return ""
            
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        
        prompt = f"""
        The user entered: "{input_text}"
        
        If this is already a valid stock ticker symbol (like AAPL, MSFT, GOOGL, BRK-A), return it in uppercase, and '.' replaced by '-'.
        If this is a company name (like Apple, Microsoft, Google, Berkshire Hathaway), return the correct stock ticker symbol.
        If the company name is not found in your knowledge base, make your best guess based on the company name pattern.
        
        Return ONLY the ticker symbol in uppercase, nothing else. No explanations, no formatting.
        
        Examples:
        - "AAPL" -> "AAPL"
        - "Apple" -> "AAPL" 
        - "Microsoft" -> "MSFT"
        - "GOOGL" -> "GOOGL"
        - "Google" -> "GOOGL"
        - "Berkshire Hathaway" -> "BRK-A"
        - "Tesla" -> "TSLA"
        - "Amazon" -> "AMZN"
        """
        
        url = f"{api_base}/models/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "text/plain"
            }
        }
        
        r = requests.post(url, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # Extract the ticker from Gemini response
        ticker = data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
        
        # Clean up common ticker formats (remove spaces, but keep hyphens for tickers like BRK-A)
        ticker = re.sub(r'[^A-Z0-9-]', '', ticker)
        
        # Validate the returned ticker - handle both regular tickers and ones with hyphens
        if re.match(r'^[A-Z0-9]{1,5}(-[A-Z0-9]{1,2})?$', ticker):
            # Test the ticker by fetching data
            try:
                test_ticker = yf.Ticker(ticker)
                test_data = test_ticker.history(period="1d")
                if not test_data.empty:
                    return ticker
            except Exception as e:
                print(f"Ticker validation failed for {ticker}: {e}")
        
        # If Gemini fails or returns invalid ticker, try web search as fallback
        try:
            print(f"Attempting web search for: {input_text}")
            search_results = ddgs_search(
                query=f"{input_text} stock ticker symbol",
                max_results=5
            )
            
            if search_results and len(search_results) > 0:
                # Extract ticker from search results
                search_text = " ".join([result.get('title', '') + ' ' + result.get('content', '') for result in search_results])
                
                # Use Gemini to parse the search results and extract the ticker
                parse_prompt = f"""
                Based on these search results about "{input_text}", extract the correct stock ticker symbol.
                
                Search results:
                {search_text}
                
                Return ONLY the ticker symbol in uppercase, nothing else. No explanations.
                Examples: "AAPL", "MSFT", "BRK-A", "GOOGL"
                """
                
                parse_body = {
                    "contents": [{"role": "user", "parts": [{"text": parse_prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "responseMimeType": "text/plain"
                    }
                }
                
                parse_r = requests.post(url, json=parse_body, timeout=30)
                parse_r.raise_for_status()
                parse_data = parse_r.json()
                
                parsed_ticker = parse_data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
                parsed_ticker = re.sub(r'[^A-Z0-9-]', '', parsed_ticker)
                
                # Validate the parsed ticker
                if re.match(r'^[A-Z0-9]{1,5}(-[A-Z0-9]{1,2})?$', parsed_ticker):
                    try:
                        test_ticker = yf.Ticker(parsed_ticker)
                        test_data = test_ticker.history(period="1d")
                        if not test_data.empty:
                            print(f"✓ Web search + Gemini parsing successful: {parsed_ticker}")
                            return parsed_ticker
                    except Exception as e:
                        print(f"Parsed ticker validation failed for {parsed_ticker}: {e}")
            else:
                print("⚠ No search results found")
                return ""  # Return empty string if no ticker found
                        
        except Exception as e:
            print(f"Web search fallback failed: {e}")
        
        # If all else fails, return empty string to indicate no ticker found
        return ""
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return ""






def load_prices(ticker: str, days: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches historical OHLC data for a ticker over N days.
    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data to fetch
    Returns:
        List of dicts: {ticker,date,open,high,low,close,volume}
    Next:
        Pass this list as price_data to compute_indicators.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
    if hist.empty:
        return []
    out = []
    for date, row in hist.iterrows():
        out.append({
            "ticker": ticker,
            "date": date,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })
    return out


def load_crypto_prices(ticker: str, days: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches historical OHLC data for a cryptocurrency ticker over N days.
    Args:
        ticker: Crypto ticker symbol (e.g., 'BTC-USD')
        days: Number of days of historical data to fetch
    Returns:
        List of dicts: {ticker,date,open,high,low,close,volume}
    Next:
        Pass this list as price_data to compute_indicators.
    """
    try:
        # Use Yahoo Finance for crypto data (same as stocks)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if hist.empty:
            return []
        out = []
        for date, row in hist.iterrows():
            out.append({
                "ticker": ticker,
                "date": date,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })
        return out
    except Exception as e:
        print(f"Error loading crypto prices for {ticker}: {e}")
        return []


def compute_indicators(price_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculates moving averages, daily returns, and volatility based on available data.
    Args:
        price_data: Output list from load_prices
    Returns:
        List of dicts: {date,ma5,ma10,daily_return,volatility}
    Next:
        Provide as indicator_data to detect_events.
    """
    if not price_data:
        return []
    df = pd.DataFrame(price_data).sort_values('date')
    df['daily_return'] = df['close'].pct_change()
    
    # Use adaptive window sizes based on available data
    n = len(df)
    ma5_window = min(5, n)
    ma10_window = min(10, n)
    vol_window = min(10, n)  # Use 10-day volatility instead of 30-day
    
    df['ma5'] = df['close'].rolling(ma5_window).mean()
    df['ma10'] = df['close'].rolling(ma10_window).mean()
    df['volatility'] = df['daily_return'].rolling(vol_window).std() * np.sqrt(252)
    
    # Drop rows with NaN values (need at least the largest window size)
    required_columns = ['ma5', 'ma10', 'daily_return', 'volatility']
    df = df.dropna(subset=required_columns)
    
    out = []
    for _, r in df.iterrows():
        out.append({
            "date": r['date'],
            "ma5": float(r['ma5']),
            "ma10": float(r['ma10']),
            "daily_return": float(r['daily_return']),
            "volatility": float(r['volatility'])
        })
    return out


def detect_events(indicator_data: List[Dict[str, Any]], threshold: float = 2.0) -> List[Dict[str, Any]]:
    """
    Flags price movements where |Δ| >= threshold%.
    Args:
        indicator_data: Output from compute_indicators
        threshold: Percentage threshold for event detection
    Returns:
        List of dicts: {date,price,change_percent,direction}
    Next:
        Provide as events to build_report and to fetch_news context.
    """
    events = []
    for r in indicator_data:
        pct = float(r['daily_return'] * 100)
        if abs(pct) >= threshold:
            events.append({
                "date": r['date'],
                "price": float(r['ma5']),
                "change_percent": pct,
                "direction": "UP" if pct > 0 else "DOWN"
            })
    return events


def forecast_prices(indicator_data: List[Dict[str, Any]], days: int = 5) -> List[Dict[str, Any]]:
    """
    Simple price forecasting based on recent trends and indicators.
    Args:
        indicator_data: Output from compute_indicators
        days: Number of days to forecast
    Returns:
        List of dicts: {date,forecast_price,confidence}
    Next:
        Provide as forecast_data to build_report.
    """
    if not indicator_data:
        return []
    
    # Get the most recent data point
    latest = indicator_data[-1] if indicator_data else {}
    if not latest:
        return []
    
    # Simple forecast based on recent trend and volatility
    latest_price = latest.get('ma5', 0)  # Use 5-day moving average as base
    daily_return = latest.get('daily_return', 0)
    volatility = latest.get('volatility', 0) / (252 ** 0.5)  # Convert to daily volatility
    
    forecasts = []
    base_date = latest.get('date', datetime.now())
    
    # Ensure base_date is a datetime object
    if isinstance(base_date, str):
        try:
            base_date = datetime.strptime(base_date, '%Y-%m-%d')
        except ValueError:
            base_date = datetime.now()
    elif not isinstance(base_date, datetime):
        base_date = datetime.now()
    
    for i in range(1, days + 1):
        # Simple forecast: base price * (1 + expected return)
        # Expected return is based on recent trend, with some randomness based on volatility
        expected_return = daily_return + np.random.normal(0, volatility * 0.1)  # Add some randomness
        forecast_price = latest_price * (1 + expected_return) ** i
        
        # Confidence decreases with forecast horizon
        confidence = max(0.5, 1.0 - (i * 0.1))
        
        forecast_date = base_date + timedelta(days=i)
        forecasts.append({
            "date": forecast_date,
            "forecast_price": float(forecast_price),
            "confidence": float(confidence),
            "trend": "UP" if expected_return > 0 else "DOWN"
        })
    
    return forecasts


def classify_asset_type(input_text: str) -> str:
    """
    Classify asset type using Gemini API to determine if input refers to a stock, cryptocurrency, or is ambiguous.
    Args:
        input_text: User input text to classify
    Returns:
        'stock', 'crypto', or 'ambiguous'
    """
    try:
        # Handle empty input
        if not input_text or not input_text.strip():
            return "ambiguous"
            
        # Use Gemini to classify the asset type
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # No fallback - require Gemini API for dynamic classification
            return "ambiguous"
            
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        
        prompt = f"""
        Analyze the following input and determine if it refers to a stock, cryptocurrency, or is ambiguous:
        
        Input: "{input_text}"
        
        Classification rules:
        - Return "stock" if the input clearly refers to a traditional stock, company, or stock ticker symbol (e.g., "AAPL", "Apple", "Microsoft", "GOOGL")
        - Return "crypto" if the input clearly refers to a cryptocurrency, crypto token, or crypto exchange (e.g., "Bitcoin", "BTC", "Ethereum", "ETH", "Dogecoin")
        - Return "ambiguous" if the input could refer to both, is unclear, or doesn't clearly match either category
        
        Return ONLY one word: "stock", "crypto", or "ambiguous". No explanations, no formatting.
        
        Examples:
        - "AAPL" -> "stock"
        - "Apple" -> "stock"
        - "Bitcoin" -> "crypto"
        - "BTC" -> "crypto"
        - "Tesla" -> "stock"
        - "Ethereum" -> "crypto"
        - "TSLA" -> "stock"
        - "ETH" -> "crypto"
        - "Something random" -> "ambiguous"
        - "XYZ" -> "ambiguous"
        """
        
        url = f"{api_base}/models/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "text/plain"
            }
        }
        
        r = requests.post(url, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # Extract the classification from Gemini response
        classification = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
        
        # Validate the response
        if classification in ["stock", "crypto", "ambiguous"]:
            return classification
        else:
            # If Gemini returns something unexpected, default to ambiguous
            return "ambiguous"
            
    except Exception as e:
        print(f"Asset classification failed: {e}")
        return "ambiguous"


def generate_analysis_summary(ticker: str, events: List[Dict[str, Any]], forecasts: List[Dict[str, Any]]) -> str:
    """
    Generate a dynamic analysis summary using Gemini.
    """
    try:
        # Create a summary of the analysis data
        event_count = len(events)
        forecast_count = len(forecasts)
        
        # Get latest forecast trend
        latest_trend = forecasts[-1].get('trend', 'NEUTRAL') if forecasts else 'NEUTRAL'
        
        # Create a prompt for Gemini
        prompt = f"""
        Based on the stock analysis for {ticker} with the following results:
        - {event_count} significant price events detected
        - {forecast_count} day price forecast available
        - Latest forecast trend: {latest_trend}
        
        Provide a concise 2-3 sentence professional analysis summary that:
        1. Highlights the key findings
        2. Mentions the forecast direction
        3. Includes appropriate disclaimers about the analysis being for informational purposes
        
        Keep it professional and concise.
        """
        
        # Use a simple approach since we don't have direct access to Gemini here
        # In a real implementation, this would call the Gemini API
        summary = f"This analysis of {ticker} identified {event_count} significant price events and generated a {forecast_count}-day forecast. "
        
        if latest_trend == "UP":
            summary += "The forecast suggests potential upward momentum, but this should be considered alongside other market factors. "
        elif latest_trend == "DOWN":
            summary += "The forecast indicates potential downward pressure, though market conditions can change rapidly. "
        else:
            summary += "The forecast shows mixed signals, suggesting a cautious approach may be warranted. "
            
        summary += "This automated analysis is for informational purposes only and should not be considered financial advice."
        
        return summary
    except Exception as e:
        # Fallback to static message if there's any error
        return "This report was generated automatically by the Agentic-Ticker analysis system. The forecasts are based on simple trend analysis and should not be considered financial advice. Please verify all information before making investment decisions."


# Update the build_report function to use dynamic conclusion
def get_company_info(ticker: str) -> Dict[str, str]:
    """
    Get company name and basic info for a ticker.
    Args:
        ticker: Stock ticker symbol
    Returns:
        Dict with company info
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "company_name": info.get('longName', ticker),
            "short_name": info.get('shortName', ticker)
        }
    except Exception as e:
        print(f"Error getting company info for {ticker}: {e}")
        return {"ticker": ticker, "company_name": ticker, "short_name": ticker}


def convert_yahoo_ticker_to_coingecko_id(yahoo_ticker: str, original_input: str = "") -> str:
    """
    Convert Yahoo Finance crypto ticker format to CoinGecko coin ID using dynamic resolution.
    Args:
        yahoo_ticker: Yahoo Finance ticker (e.g., 'DOGE-USD', 'BTC-USD')
        original_input: Original user input (e.g., 'PHALA NETWORK', 'DOGECOIN')
    Returns:
        CoinGecko coin ID (e.g., 'dogecoin', 'bitcoin', 'phala-network')
    """
    # Read Gemini configuration up-front
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    api_base = os.getenv("GEMINI_API_BASE")
    # URL will be built when api_key is present
    if api_key:
        url = f"{api_base}/models/{model}:generateContent?key={api_key}"
    else:
        url = None

    # Remove the -USD suffix
    if yahoo_ticker.endswith('-USD'):
        base_ticker = yahoo_ticker[:-4]
    else:
        base_ticker = yahoo_ticker

    # Use original input if available, otherwise use base ticker
    search_term = original_input if original_input else base_ticker

    # Try to get coin ID using Gemini first
    try:
        if api_key:
            prompt = f"""
            Convert this cryptocurrency name or ticker to the correct CoinGecko coin ID:

            Input: "{search_term}"

            CoinGecko uses specific coin IDs (like 'bitcoin', 'ethereum', 'dogecoin', 'pha') 
            rather than ticker symbols. These are usually short, lowercase identifiers.

            Return ONLY the CoinGecko coin ID in lowercase, nothing else. No explanations.

            Examples:
            - "Bitcoin" -> "bitcoin"
            - "BTC" -> "bitcoin" 
            - "Dogecoin" -> "dogecoin"
            - "DOGE" -> "dogecoin"
            - "Ethereum" -> "ethereum"
            - "PHALA NETWORK" -> "pha"
            - "PHA" -> "pha"
            """

            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "text/plain"
                }
            }

            r = requests.post(url, json=body, timeout=30)
            r.raise_for_status()
            data = r.json()

            # Extract the coin ID from Gemini response
            coin_id = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()

            # Clean up coin ID (remove spaces, special characters except hyphens)
            coin_id = re.sub(r'[^a-z0-9-]', '', coin_id)

            if coin_id and len(coin_id) > 2:
                print(f"✓ Gemini resolved '{search_term}' to CoinGecko ID: '{coin_id}'")
                return coin_id

    except Exception as e:
        print(f"Gemini coin ID resolution failed for {search_term}: {e}")

    # Fallback: Try web search
    try:
        print(f"Attempting web search for coin ID: {search_term}")
        search_results = ddgs_search(
            query=f"{search_term} CoinGecko coin ID cryptocurrency",
            max_results=3
        )

        if search_results and len(search_results) > 0:
            # Extract coin ID from search results
            search_text = " ".join([result.get('title', '') + ' ' + result.get('content', '') for result in search_results])

            # Use Gemini to parse the search results and extract the coin ID
            parse_prompt = f"""
            Based on these search results about "{search_term}", extract the correct CoinGecko coin ID.

            Search results:
            {search_text}

            Return ONLY the CoinGecko coin ID in lowercase, nothing else. No explanations.
            Examples: "bitcoin", "ethereum", "dogecoin", "phala-network"
            """

            if api_key:
                parse_body = {
                    "contents": [{"role": "user", "parts": [{"text": parse_prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "responseMimeType": "text/plain"
                    }
                }

                parse_r = requests.post(url, json=parse_body, timeout=30)
                parse_r.raise_for_status()
                parse_data = parse_r.json()

                parsed_coin_id = parse_data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
                parsed_coin_id = re.sub(r'[^a-z0-9-]', '', parsed_coin_id)

                if parsed_coin_id and len(parsed_coin_id) > 2:
                    print(f"✓ Web search + Gemini parsing resolved '{search_term}' to CoinGecko ID: '{parsed_coin_id}'")
                    return parsed_coin_id

    except Exception as e:
        print(f"Web search coin ID resolution failed for {search_term}: {e}")

    # Final fallback: return empty string if all resolution methods fail
    print(f"⚠ All resolution methods failed for {search_term}")
    return ""


def get_crypto_info(ticker: str, original_input: str = "") -> Dict[str, Any]:
    """
    Get basic information about a cryptocurrency using CoinGecko API.
    Args:
        ticker: Cryptocurrency ticker symbol (e.g., 'bitcoin', 'ethereum', 'btc')
    Returns:
        Dict with cryptocurrency info including name, symbol, market cap, etc.
    """
    # Read API keys / env vars up-front
    demo_api_key = os.getenv("COINGECKO_DEMO_API_KEY")
    pro_api_key = os.getenv("COINGECKO_API_KEY")

    try:
        if not CoinGeckoAPI:
            raise ImportError("CoinGeckoAPI not available")

        # Initialize client based on available keys (demo > pro > default)
        if demo_api_key:
            client = CoinGeckoAPI(demo_api_key=demo_api_key, environment="demo")
        elif pro_api_key:
            client = CoinGeckoAPI(pro_api_key=pro_api_key, environment="pro")
        else:
            client = CoinGeckoAPI()

        # Convert Yahoo Finance ticker to CoinGecko coin ID
        # CoinGecko uses coin IDs (like 'bitcoin') rather than symbols
        coin_id = convert_yahoo_ticker_to_coingecko_id(ticker, original_input)

        # Get coin data
        coin_data = client.coins.get_id(id=coin_id)

        # Extract data with proper attribute access
        market_data = getattr(coin_data, 'market_data', None)

        current_price_usd = 0
        market_cap_usd = 0
        total_volume_usd = 0
        price_change_percentage_24h = 0
        price_change_percentage_7d = 0
        circulating_supply = 0
        total_supply = 0

        if market_data:
            # Access nested attributes properly
            current_price = getattr(market_data, 'current_price', None)
            if current_price and hasattr(current_price, 'usd'):
                current_price_usd = getattr(current_price, 'usd', 0)

            market_cap = getattr(market_data, 'market_cap', None)
            if market_cap and hasattr(market_cap, 'usd'):
                market_cap_usd = getattr(market_cap, 'usd', 0)

            total_volume = getattr(market_data, 'total_volume', None)
            if total_volume and hasattr(total_volume, 'usd'):
                total_volume_usd = getattr(total_volume, 'usd', 0)

            price_change_percentage_24h = getattr(market_data, 'price_change_percentage_24h', 0)
            price_change_percentage_7d = getattr(market_data, 'price_change_percentage_7d', 0)
            circulating_supply = getattr(market_data, 'circulating_supply', 0)
            total_supply = getattr(market_data, 'total_supply', 0)

        return {
            "ticker": ticker.upper(),
            "coin_id": coin_id,
            "name": getattr(coin_data, 'name', ticker),
            "symbol": getattr(coin_data, 'symbol', ticker).upper(),
            "market_cap_usd": market_cap_usd,
            "current_price_usd": current_price_usd,
            "price_change_percentage_24h": price_change_percentage_24h,
            "price_change_percentage_7d": price_change_percentage_7d,
            "total_volume_usd": total_volume_usd,
            "circulating_supply": circulating_supply,
            "total_supply": total_supply,
            "last_updated": datetime.now()
        }

    except Exception as e:
        print(f"Error getting crypto info for {ticker}: {e}")

        # Try to search for the coin if direct lookup fails
        try:
            if CoinGeckoAPI:
                # Reuse previously-read keys to initialize client for search fallback
                if pro_api_key:
                    client = CoinGeckoAPI(pro_api_key=pro_api_key, environment="pro")
                elif demo_api_key:
                    client = CoinGeckoAPI(demo_api_key=demo_api_key, environment="demo")
                else:
                    client = CoinGeckoAPI()

                search_results = client.search.get(query=original_input if original_input else ticker)
                coins = getattr(search_results, 'coins', [])

                if coins:
                    # Use the first result
                    first_coin = coins[0]
                    coin_id = getattr(first_coin, 'id', None)

                    if coin_id:
                        coin_data = client.coins.get_id(id=coin_id)

                        # Extract data with proper attribute access
                        market_data = getattr(coin_data, 'market_data', None)

                        current_price_usd = 0
                        market_cap_usd = 0
                        total_volume_usd = 0
                        price_change_percentage_24h = 0
                        price_change_percentage_7d = 0
                        circulating_supply = 0
                        total_supply = 0

                        if market_data:
                            # Access nested attributes properly
                            current_price = getattr(market_data, 'current_price', None)
                            if current_price and hasattr(current_price, 'usd'):
                                current_price_usd = getattr(current_price, 'usd', 0)

                            market_cap = getattr(market_data, 'market_cap', None)
                            if market_cap and hasattr(market_cap, 'usd'):
                                market_cap_usd = getattr(market_cap, 'usd', 0)

                            total_volume = getattr(market_data, 'total_volume', None)
                            if total_volume and hasattr(total_volume, 'usd'):
                                total_volume_usd = getattr(total_volume, 'usd', 0)

                            price_change_percentage_24h = getattr(market_data, 'price_change_percentage_24h', 0)
                            price_change_percentage_7d = getattr(market_data, 'price_change_percentage_7d', 0)
                            circulating_supply = getattr(market_data, 'circulating_supply', 0)
                            total_supply = getattr(market_data, 'total_supply', 0)

                        return {
                            "ticker": ticker.upper(),
                            "coin_id": coin_id,
                            "name": getattr(coin_data, 'name', ticker),
                            "symbol": getattr(coin_data, 'symbol', ticker).upper(),
                            "market_cap_usd": market_cap_usd,
                            "current_price_usd": current_price_usd,
                            "price_change_percentage_24h": price_change_percentage_24h,
                            "price_change_percentage_7d": price_change_percentage_7d,
                            "total_volume_usd": total_volume_usd,
                            "circulating_supply": circulating_supply,
                            "total_supply": total_supply,
                            "last_updated": datetime.now()
                        }

        except Exception as search_e:
            print(f"Crypto search fallback failed for {ticker}: {search_e}")

        # Return minimal info if all attempts fail
        return {
            "ticker": ticker.upper(),
            "coin_id": ticker.lower(),
            "name": ticker,
            "symbol": ticker.upper(),
            "market_cap_usd": 0,
            "current_price_usd": 0,
            "price_change_percentage_24h": 0,
            "price_change_percentage_7d": 0,
            "total_volume_usd": 0,
            "circulating_supply": 0,
            "total_supply": 0,
            "last_updated": datetime.now(),
            "error": f"Failed to fetch crypto data for {ticker}"
        }


def build_report(ticker: str, events: List[Dict[str, Any]], forecasts: List[Dict[str, Any]], company_info: Optional[Dict[str, str]] = None, crypto_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generates a markdown brief with events and price forecasts with enhanced formatting and colors.
    Args:
        ticker: Stock ticker symbol
        events: Output from detect_events
        forecasts: Output from forecast_prices
    Returns:
        Dict with keys {ticker,analysis_period,generated_date,content}
    """
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    # Use company name or crypto name if available, otherwise use ticker
    if company_info:
        display_name = f"{company_info.get('company_name', ticker)} ({ticker})"
        report_type = "Stock"
    elif crypto_info:
        display_name = f"{crypto_info.get('name', ticker)} ({ticker})"
        report_type = "Cryptocurrency"
    else:
        display_name = ticker
        report_type = "Asset"
    
    md = [f"# 📊 {report_type} Analysis Report for {display_name}", "", f"**Analysis Period**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"]
    md.append("")
    
    # Add crypto information if available
    if crypto_info:
        md.append("## ₿ Cryptocurrency Information")
        md.append(f"| **Symbol** | {crypto_info.get('symbol', 'N/A')}")
        md.append(f"| **Current Price** | ${crypto_info.get('current_price_usd', 0):,.2f}")
        md.append(f"| **Market Cap** | ${crypto_info.get('market_cap_usd', 0):,.0f}")
        md.append(f"| **24h Change** | {crypto_info.get('price_change_percentage_24h', 0):+.2f}%")
        md.append(f"| **7d Change** | {crypto_info.get('price_change_percentage_7d', 0):+.2f}%")
        md.append(f"| **24h Volume** | ${crypto_info.get('total_volume_usd', 0):,.0f}")
        md.append("")
    
    # Add company information if available
    if company_info:
        md.append("## 🏢 Company Information")
        md.append(f"| **Company Name** | {company_info.get('company_name', 'N/A')}")
        md.append(f"| **Short Name** | {company_info.get('short_name', 'N/A')}")
        md.append("")
    
    md.append("## 📈 Significant Price Events")
    if events:
        md.append("| Date | Price | Change | Direction |")
        md.append("|------|-------|--------|-----------|")
        for ev in events:
            d = ev.get("date")
            # Format date properly - extract just the date part
            if d is None:
                formatted_date = "N/A"
            elif hasattr(d, 'strftime'):
                # Handle pandas Timestamp and datetime objects
                try:
                    formatted_date = d.strftime('%Y-%m-%d')
                except (ValueError, TypeError, AttributeError):
                    # Fallback for pandas Timestamp with timezone
                    if hasattr(d, 'normalize'):
                        formatted_date = d.normalize().strftime('%Y-%m-%d')
                    elif hasattr(d, 'date'):
                        formatted_date = d.date().strftime('%Y-%m-%d')
                    else:
                        formatted_date = str(d).split(' ')[0]  # Get first part (date)
            elif isinstance(d, str):
                # Handle string dates - split on space or T to remove time part
                formatted_date = d.split(' ')[0].split('T')[0]
            elif hasattr(d, 'isoformat'):
                formatted_date = d.isoformat().split('T')[0]  # Get just the date part
            else:
                formatted_date = str(d).split(' ')[0]  # Get first part (date)
            
            p = ev.get("price", 0.0)
            c = ev.get("change_percent", 0.0)
            dr = ev.get("direction", "")
            # Add color coding for direction
            if dr == "UP":
                direction = f"🟢 {dr}"
            elif dr == "DOWN":
                direction = f"🔴 {dr}"
            else:
                direction = dr
            # Add color coding for change
            change_color = "📈" if c > 0 else "📉" if c < 0 else "➡️"
            md.append(f"| {formatted_date} | ${p:.2f} | {change_color} {c:+.2f}% | {direction} |")
    else:
        md.append("📋 No significant price events detected.")
    md.append("")
    md.append("## 🔮 Price Forecasts")
    if forecasts:
        md.append("| Date | Forecast Price | Confidence | Trend |")
        md.append("|------|----------------|------------|-------|")
        for f in forecasts:
            d = f.get("date")
            # Format date properly - extract just the date part
            if d is None:
                formatted_date = "N/A"
            elif hasattr(d, 'strftime'):
                # Handle pandas Timestamp and datetime objects
                try:
                    formatted_date = d.strftime('%Y-%m-%d')
                except (ValueError, TypeError, AttributeError):
                    # Fallback for pandas Timestamp with timezone
                    if hasattr(d, 'normalize'):
                        formatted_date = d.normalize().strftime('%Y-%m-%d')
                    elif hasattr(d, 'date'):
                        formatted_date = d.date().strftime('%Y-%m-%d')
                    else:
                        formatted_date = str(d).split(' ')[0]  # Get first part (date)
            elif isinstance(d, str):
                # Handle string dates - split on space or T to remove time part
                formatted_date = d.split(' ')[0].split('T')[0]
            elif hasattr(d, 'isoformat'):
                formatted_date = d.isoformat().split('T')[0]  # Get just the date part
            else:
                formatted_date = str(d).split(' ')[0]  # Get first part (date)
            price = f.get("forecast_price", 0.0)
            conf = f.get("confidence", 0.0) * 100
            trend = f.get("trend", "")
            # Add color coding for confidence and trend
            if conf >= 80:
                conf_emoji = "🟢"
            elif conf >= 60:
                conf_emoji = "🟡"
            else:
                conf_emoji = "🔴"
            
            if trend == "UP":
                trend_emoji = "📈"
            elif trend == "DOWN":
                trend_emoji = "📉"
            else:
                trend_emoji = "➡️"
            
            md.append(f"| {formatted_date} | ${price:.2f} | {conf_emoji} {conf:.1f}% | {trend_emoji} {trend} |")
    else:
        md.append("📋 No price forecasts available.")
    md.append("")
    md.append("## 🎯 Conclusion")
    summary = generate_analysis_summary(ticker, events, forecasts)
    md.append(summary)
    md.append("")
    md.append("---")
    md.append("*🤖 Report generated by Agentic-Ticker | Last updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*")
    return {
        "ticker": ticker,
        "analysis_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "generated_date": datetime.now(),
        "content": "\n".join(md)
    }


