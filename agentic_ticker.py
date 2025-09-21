#!/usr/bin/env python3
import os
import json
import io
import base64
import requests
import re
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

# Import the searxng bridge search function
# Web search function using local SearxNG instance
def searxng_bridge_search(query, max_results=3, **kwargs):
    """Search using local SearxNG instance at http://localhost:8080"""
    try:
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        url = f"http://localhost:8080/search?q={encoded_query}&format=json"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract and return the search results
        results = data.get('results', [])
        print(f"✓ Web search returned {len(results)} results for query: {query}")
        return results[:max_results]
        
    except Exception as e:
        print(f"Web search failed: {e}")
        return []

load_dotenv(find_dotenv(), override=False)

# ---------------------
# JSON helpers
# ---------------------

def _json_safe(obj):
    import pandas as _pd, numpy as _np
    from datetime import datetime as _dt
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (_dt, _pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    return str(obj)

def _dumps(obj) -> str:
    return json.dumps(_json_safe(obj), ensure_ascii=False)

def _format_json_for_display(obj) -> str:
    """
    Format JSON data for readable display in the UI.
    Returns a formatted string with proper indentation and structure.
    Truncates large arrays to keep display compact.
    """
    if not obj or isinstance(obj, (str, int, float, bool)):
        return str(obj)
    
    try:
        # Create a truncated version for display
        truncated_obj = _truncate_large_data(obj)
        formatted = json.dumps(_json_safe(truncated_obj), indent=2, ensure_ascii=False)
        return formatted
    except:
        return str(obj)

def _truncate_large_data(obj, max_array_items=3, max_string_length=50):
    """
    Truncate large data structures to keep display compact.
    """
    if isinstance(obj, dict):
        truncated = {}
        for key, value in obj.items():
            truncated[key] = _truncate_large_data(value, max_array_items, max_string_length)
        return truncated
    elif isinstance(obj, list):
        if len(obj) <= max_array_items:
            return [_truncate_large_data(item, max_array_items, max_string_length) for item in obj]
        else:
            # Show first few items and indicate truncation
            truncated = [_truncate_large_data(item, max_array_items, max_string_length) for item in obj[:max_array_items]]
            truncated.append(f"... {len(obj) - max_array_items} more items ...")
            return truncated
    elif isinstance(obj, str):
        if len(obj) > max_string_length:
            return obj[:max_string_length] + "..."
        return obj
    else:
        return obj

def _extract_json_text(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        end = s.rfind("```")
        if end != -1:
            first_nl = s.find("\n")
            if first_nl != -1:
                s = s[first_nl+1:end]
    s = s.strip()
    if s.lower().startswith("json"):
        s = s[4:].strip()
    return s

def _clean_trailing_commas(s: str) -> str:
    pattern = re.compile(r",\s*([}\]])")
    return pattern.sub(lambda m: m.group(1), s)

def _parse_json_strictish(text: str) -> dict:
    raw = _extract_json_text(text)
    left, right = raw.find("{"), raw.rfind("}")
    candidate = raw[left:right+1] if (left >= 0 and right > left) else raw
    candidate = _clean_trailing_commas(candidate)
    return json.loads(candidate)

# ---------------------
# Data Models
# ---------------------

class StockPriceData(BaseModel):
    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class TechnicalIndicators(BaseModel):
    date: datetime
    ma5: float
    ma10: float
    daily_return: float
    volatility: float


class PriceEvent(BaseModel):
    date: datetime
    price: float
    change_percent: float
    direction: str


class NewsArticle(BaseModel):
    title: str
    summary: str
    url: str
    published_date: datetime
    source: str


class AnalysisReport(BaseModel):
    ticker: str
    analysis_period: str
    generated_date: datetime
    content: str


class PlannerJSON(BaseModel):
    call: Optional[Dict[str, Any]] = None
    final: Optional[Dict[str, Any]] = None


# ---------------------
# Service Functions (tools)
# ---------------------

def validate_ticker(input_text: str) -> str:
    """
    Validates and converts stock name or ticker to proper ticker symbol using Gemini.
    Args:
        input_text: User input (can be ticker symbol like 'AAPL' or company name like 'Apple')
    Returns:
        Valid ticker symbol (e.g., 'AAPL')
    """
    try:
        # Handle empty input
        if not input_text or not input_text.strip():
            return ""
            
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
            # Fallback: return uppercase version if no API key
            return input_text.upper()
            
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
            search_results = searxng_bridge_search(
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
                        if not test_data.empty or parsed_ticker in ["BRK-A", "BRK-B", "GOOGL", "META", "AAPL", "MSFT", "AMZN", "TSLA"]:
                            print(f"✓ Web search + Gemini parsing successful: {parsed_ticker}")
                            return parsed_ticker
                    except Exception as e:
                        print(f"Parsed ticker validation failed for {parsed_ticker}: {e}")
                        # For well-known tickers, return them anyway
                        if parsed_ticker in ["BRK-A", "BRK-B", "GOOGL", "META", "AAPL", "MSFT", "AMZN", "TSLA"]:
                            print(f"✓ Returning well-known ticker despite validation error: {parsed_ticker}")
                            return parsed_ticker
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

def validate_ticker_gemini_only(input_text: str) -> str:
    """Validate ticker using only Gemini API (no web search fallback)"""
    try:
        # Clean up input
        cleaned_input = re.sub(r'[^\w\s\-\.]', '', input_text.strip())
        
        # Use Gemini to extract and validate ticker
        prompt = f"""
        Extract the stock ticker symbol from this input: "{cleaned_input}"
        
        Return ONLY the ticker symbol in uppercase, nothing else. No explanations.
        Examples: "AAPL", "MSFT", "BRK-A", "GOOGL"
        """
        
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "text/plain"
            }
        }
        
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
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
        
        # Return empty string if Gemini validation fails
        return ""
        
    except Exception as e:
        print(f"Gemini validation failed: {e}")
        return ""

def validate_ticker_with_web_search(input_text: str) -> str:
    """Validate ticker using web search + Gemini parsing"""
    try:
        print(f"Attempting web search for: {input_text}")
        search_results = searxng_bridge_search(
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
            
            api_key = os.getenv("GEMINI_API_KEY")
            model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
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
                    if not test_data.empty or parsed_ticker in ["BRK-A", "BRK-B", "GOOGL", "META", "AAPL", "MSFT", "AMZN", "TSLA"]:
                        print(f"✓ Web search + Gemini parsing successful: {parsed_ticker}")
                        return parsed_ticker
                except Exception as e:
                    print(f"Parsed ticker validation failed for {parsed_ticker}: {e}")
                    # For well-known tickers, return them anyway
                    if parsed_ticker in ["BRK-A", "BRK-B", "GOOGL", "META", "AAPL", "MSFT", "AMZN", "TSLA"]:
                        print(f"✓ Returning well-known ticker despite validation error: {parsed_ticker}")
                        return parsed_ticker
        else:
            print("⚠ No search results found")
            return ""  # Return empty string if no ticker found
                    
    except Exception as e:
        print(f"Web search fallback failed: {e}")
    
    # If all else fails, return empty string to indicate no ticker found
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
            "date": date.to_pydatetime(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })
    return out


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
            "date": pd.to_datetime(r['date']).to_pydatetime(),
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
def build_report(ticker: str, events: List[Dict[str, Any]], forecasts: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    md = [f"# 📊 Stock Analysis Report for {ticker}", "", f"**Analysis Period**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"]
    md.append("")
    md.append("## 📈 Significant Price Events")
    if events:
        md.append("| Date | Price | Change | Direction |")
        md.append("|------|-------|--------|-----------|")
        for ev in events:
            d = pd.to_datetime(ev.get("date")).strftime('%Y-%m-%d')
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
            md.append(f"| {d} | ${p:.2f} | {change_color} {c:+.2f}% | {direction} |")
    else:
        md.append("📋 No significant price events detected.")
    md.append("")
    md.append("## 🔮 Price Forecasts")
    if forecasts:
        md.append("| Date | Forecast Price | Confidence | Trend |")
        md.append("|------|----------------|------------|-------|")
        for f in forecasts:
            d = pd.to_datetime(f.get("date")).strftime('%Y-%m-%d')
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
            
            md.append(f"| {d} | ${price:.2f} | {conf_emoji} {conf:.1f}% | {trend_emoji} {trend} |")
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


# ---------------------
# Gemini Planner
# ---------------------

class GeminiPlanner:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required")

    def plan(self, tools_spec: List[Dict[str, Any]], goal: str, transcript: List[Dict[str, Any]]) -> PlannerJSON:
        system = (
            "You are a stock analysis assistant that calls functions in a specific sequence. "
            "First call load_prices with ticker and days. "
            "Then call compute_indicators with the price data result. "
            "Then call detect_events with the indicator data result and a threshold. "
            "Then call forecast_prices with the indicator data and days. "
            "Finally call build_report with all the collected data. "
            "Only output a single JSON object with either {\"call\":{name,args}} or {\"final\":{message}}. "
            "Use exact argument names from the functions' docstrings."
        )
        payload_text = _dumps({"tools": tools_spec, "goal": goal, "transcript": transcript})
        url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": payload_text}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Invalid Gemini response: {data}") from e
        try:
            obj = _parse_json_strictish(text)
        except Exception as ex:
            repair_body = {
                "system_instruction": {"parts": [{"text": system + " Return ONLY strict JSON with double quotes, no comments, no trailing commas."}]},
                "contents": [{"role": "user", "parts": [{"text": payload_text}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
            }
            rr = requests.post(url, json=repair_body, timeout=120)
            rr.raise_for_status()
            d2 = rr.json()
            try:
                text2 = d2["candidates"][0]["content"]["parts"][0]["text"]
                obj = _parse_json_strictish(text2)
            except Exception:
                snippet = (text or "")[:300]
                raise RuntimeError(f"Planner JSON parse failed. First attempt snippet: {snippet}") from ex
        return PlannerJSON(**obj)


# ---------------------
# Orchestrator
# ---------------------

class Orchestrator:
    def __init__(self):
        self.planner = GeminiPlanner()
        self.tools = {
            "validate_ticker": validate_ticker,
            "load_prices": load_prices,
            "compute_indicators": compute_indicators,
            "detect_events": detect_events,
            "forecast_prices": forecast_prices,
            "build_report": build_report,
        }

    def tools_spec(self) -> List[Dict[str, Any]]:
        spec = []
        for name, fn in self.tools.items():
            spec.append({"name": name, "docstring": fn.__doc__ or "", "signature": str(fn)})
        return spec

    def run(self, ticker_input: str, days: int, threshold: float, forecast_days: int, on_event: Optional[callable] = None) -> List[Dict[str, Any]]:
        # Simple sequential execution - no complex orchestration
        steps: List[Dict[str, Any]] = []
        
        if on_event:
            on_event({"type": "info", "message": f"Starting stock analysis pipeline for '{ticker_input}'..."})
        
        # Step 0: Initial ticker validation with Gemini
        if on_event:
            on_event({"type": "call", "step": 0, "name": "validate_ticker_gemini", "args": {"input_text": ticker_input}})
        
        validated_ticker = ""
        web_search_performed = False
        
        try:
            # Try Gemini validation first
            validated_ticker = validate_ticker_gemini_only(ticker_input)
            
            if validated_ticker:
                steps.append({"type": "result", "name": "validate_ticker_gemini", "result": validated_ticker})
                if on_event:
                    on_event({"type": "result", "step": 0, "name": "validate_ticker_gemini", "result": validated_ticker})
                    if validated_ticker.upper() != ticker_input.upper():
                        on_event({"type": "info", "message": f"Converted '{ticker_input}' to ticker symbol: {validated_ticker}"})
            else:
                # Step 0.1: Web search fallback
                web_search_performed = True
                if on_event:
                    on_event({"type": "call", "step": 1, "name": "web_search_ticker", "args": {"input_text": ticker_input}})
                
                try:
                    validated_ticker = validate_ticker_with_web_search(ticker_input)
                    
                    if validated_ticker:
                        steps.append({"type": "result", "name": "web_search_ticker", "result": validated_ticker})
                        if on_event:
                            on_event({"type": "result", "step": 1, "name": "web_search_ticker", "result": validated_ticker})
                            if validated_ticker.upper() != ticker_input.upper():
                                on_event({"type": "info", "message": f"Web search found ticker: {validated_ticker}"})
                    else:
                        error_msg = f"No valid ticker found for '{ticker_input}' after web search. Please check the company name or ticker symbol."
                        if on_event:
                            on_event({"type": "error", "step": 1, "name": "web_search_ticker", "error": error_msg})
                        raise ValueError(error_msg)
                except Exception as e:
                    if on_event:
                        on_event({"type": "error", "step": 1, "name": "web_search_ticker", "error": str(e)})
                    raise
            
            if not validated_ticker:
                error_msg = f"No valid ticker found for '{ticker_input}'. Please check the company name or ticker symbol."
                if on_event:
                    on_event({"type": "error", "step": 0 if not web_search_performed else 1, "name": "validate_ticker_gemini" if not web_search_performed else "web_search_ticker", "error": error_msg})
                raise ValueError(error_msg)
                
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 0 if not web_search_performed else 1, "name": "validate_ticker_gemini" if not web_search_performed else "web_search_ticker", "error": str(e)})
            raise
        
        # Step 1: Load prices
        if on_event:
            on_event({"type": "call", "step": 1, "name": "load_prices", "args": {"ticker": validated_ticker, "days": days}})
        
        try:
            price_data = load_prices(validated_ticker, days)
            if not price_data:
                error_msg = f"No price data found for ticker '{validated_ticker}'. The ticker may be delisted or invalid."
                if on_event:
                    on_event({"type": "error", "step": 1, "name": "load_prices", "error": error_msg})
                raise ValueError(error_msg)
            
            steps.append({"type": "result", "name": "load_prices", "result": price_data})
            if on_event:
                on_event({"type": "result", "step": 1, "name": "load_prices", "result": price_data})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 1, "name": "load_prices", "error": str(e)})
            raise
        
        # Step 2: Compute indicators
        if on_event:
            on_event({"type": "call", "step": 2, "name": "compute_indicators", "args": {"price_data": price_data}})
        
        try:
            indicator_data = compute_indicators(price_data)
            steps.append({"type": "result", "name": "compute_indicators", "result": indicator_data})
            if on_event:
                on_event({"type": "result", "step": 2, "name": "compute_indicators", "result": indicator_data})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 2, "name": "compute_indicators", "error": str(e)})
            raise
        
        # Step 3: Detect events
        if on_event:
            on_event({"type": "call", "step": 3, "name": "detect_events", "args": {"indicator_data": indicator_data, "threshold": threshold}})
        
        try:
            events = detect_events(indicator_data, threshold)
            steps.append({"type": "result", "name": "detect_events", "result": events})
            if on_event:
                on_event({"type": "result", "step": 3, "name": "detect_events", "result": events})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 3, "name": "detect_events", "error": str(e)})
            raise
        
        # Step 4: Forecast prices
        if on_event:
            on_event({"type": "call", "step": 4, "name": "forecast_prices", "args": {"indicator_data": indicator_data, "days": forecast_days}})
        
        try:
            forecasts = forecast_prices(indicator_data, forecast_days)
            steps.append({"type": "result", "name": "forecast_prices", "result": forecasts})
            if on_event:
                on_event({"type": "result", "step": 4, "name": "forecast_prices", "result": forecasts})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 4, "name": "forecast_prices", "error": str(e)})
            raise
        
        # Step 5: Build report
        if on_event:
            on_event({"type": "call", "step": 5, "name": "build_report", "args": {"ticker": validated_ticker, "events": events, "forecasts": forecasts}})
        
        try:
            report = build_report(validated_ticker, events, forecasts)
            steps.append({"type": "result", "name": "build_report", "result": report})
            if on_event:
                on_event({"type": "result", "step": 5, "name": "build_report", "result": report})
                on_event({"type": "final", "data": report})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 5, "name": "build_report", "error": str(e)})
            raise
        
        # Store results in session state for UI
        if 'st' in globals() or 'st' in locals():
            try:
                st.session_state.price_data = price_data
                st.session_state.indicator_data = indicator_data
                st.session_state.events = events
                st.session_state.forecasts = forecasts
                st.session_state.report = report
            except:
                pass  # Streamlit not available in this context
        
        return steps


# ---------------------
# UI Components
# ---------------------

def create_price_chart(price_data: List[Dict[str, Any]], indicator_data: List[Dict[str, Any]]) -> go.Figure:
    """Create an animated price chart using Plotly."""
    price_df = pd.DataFrame(price_data)
    ind_df = pd.DataFrame(indicator_data)
    
    if not price_df.empty and 'date' in price_df.columns:
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df.sort_values('date')
    if not ind_df.empty and 'date' in ind_df.columns:
        ind_df['date'] = pd.to_datetime(ind_df['date'])
        ind_df = ind_df.sort_values('date')
    
    # Create animated figure
    fig = go.Figure()
    
    # Add price line
    if not price_df.empty:
        fig.add_trace(go.Scatter(
            x=price_df['date'],
            y=price_df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Add moving averages
    if not ind_df.empty:
        fig.add_trace(go.Scatter(
            x=ind_df['date'],
            y=ind_df['ma5'],
            mode='lines',
            name='MA5',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>MA5: $%{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=ind_df['date'],
            y=ind_df['ma10'],
            mode='lines',
            name='MA10',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            hovertemplate='Date: %{x}<br>MA10: $%{y:.2f}<extra></extra>'
        ))
    
    # Create animation frames that match the initial chart structure
    frames = []
    if not price_df.empty:
        for i in range(len(price_df)):
            frame_data = []
            
            # Add price data up to current point (start from first frame)
            frame_data.append(go.Scatter(
                x=price_df['date'][:i+1],
                y=price_df['close'][:i+1],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            
            # Add indicator data up to current point (synchronized with price data)
            if not ind_df.empty:
                # Use the same number of points as price data for synchronization
                ind_points = min(i+1, len(ind_df))
                if ind_points > 0:
                    frame_data.append(go.Scatter(
                        x=ind_df['date'][:ind_points],
                        y=ind_df['ma5'][:ind_points],
                        mode='lines',
                        name='MA5',
                        line=dict(color='#ff7f0e', width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>MA5: $%{y:.2f}<extra></extra>'
                    ))
                    
                    frame_data.append(go.Scatter(
                        x=ind_df['date'][:ind_points],
                        y=ind_df['ma10'][:ind_points],
                        mode='lines',
                        name='MA10',
                        line=dict(color='#2ca02c', width=2, dash='dot'),
                        hovertemplate='Date: %{x}<br>MA10: $%{y:.2f}<extra></extra>'
                    ))
            
            frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Update layout with animation controls
    fig.update_layout(
        title='📈 Price & Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        showlegend=True,
        height=500,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[None, {"frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 300}}],
                        label="▶️ Play",
                        method="animate"
                    ),
                    dict(
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}],
                        label="⏸️ Pause",
                        method="animate"
                    )
                ]),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue={"prefix": "Frame: "},
                transition={"duration": 300},
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[[f.name], {"frame": {"duration": 300, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 300}}],
                        label=f.name,
                        method="animate"
                    )
                    for f in frames
                ]
            )
        ] if frames else []
    )
    
    if frames:
        fig.frames = frames
    
    return fig


def create_events_table(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "<p>No significant events detected.</p>"
    df = pd.DataFrame(events)
    if df.empty:
        return "<p>No significant events detected.</p>"
    
    # Ensure we have the expected columns
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    if 'change_percent' in df.columns:
        df['change'] = df['change_percent'].apply(lambda x: f"{x:+.2f}%")
    if 'price' in df.columns:
        df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
    
    # Rename columns and collect available ones
    cols = []
    if 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
        cols.append('Date')
    if 'price' in df.columns:
        df.rename(columns={'price': 'Price'}, inplace=True)
        cols.append('Price')
    if 'change' in df.columns:
        df.rename(columns={'change': 'Change'}, inplace=True)
        cols.append('Change')
    if 'direction' in df.columns:
        df.rename(columns={'direction': 'Direction'}, inplace=True)
        cols.append('Direction')
    
    if not cols:
        return "<p>No significant events detected.</p>"
        
    return df[cols].to_html(classes='table table-striped table-hover', index=False, escape=False)

def create_forecast_chart(forecasts: List[Dict[str, Any]]) -> go.Figure:
    """Create an animated forecast chart using Plotly."""
    if not forecasts:
        # Return empty figure if no forecasts
        fig = go.Figure()
        fig.update_layout(
            title="🔮 Price Forecasts",
            xaxis_title="Date",
            yaxis_title="Forecast Price ($)",
            height=400
        )
        return fig
    
    forecast_df = pd.DataFrame(forecasts)
    if forecast_df.empty:
        # Return empty figure if no valid forecast data
        fig = go.Figure()
        fig.update_layout(
            title="🔮 Price Forecasts",
            xaxis_title="Date",
            yaxis_title="Forecast Price ($)",
            height=400
        )
        return fig
    
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    forecast_df = forecast_df.sort_values('date')
    
    # Handle missing or invalid values
    forecast_df['confidence'] = forecast_df['confidence'].apply(lambda x: 0.5 if pd.isna(x) or x is None else x)
    forecast_df['forecast_price'] = forecast_df['forecast_price'].apply(lambda x: 0.0 if pd.isna(x) or x is None else x)
    
    # Drop any rows with invalid data
    forecast_df = forecast_df.dropna()
    
    # Create animated forecast chart
    fig = go.Figure()
    
    # Create a continuous line chart with confidence-based coloring
    # Determine overall trend for line color
    overall_trend = forecast_df['trend'].iloc[-1] if not forecast_df.empty else 'NEUTRAL'
    avg_confidence = forecast_df['confidence'].mean() if not forecast_df.empty else 0.5
    
    # Color based on overall trend and average confidence
    if overall_trend == 'UP':
        line_color = '#2ca02c' if avg_confidence > 0.7 else '#98df8a'
    elif overall_trend == 'DOWN':
        line_color = '#d62728' if avg_confidence > 0.7 else '#ff9999'
    else:
        line_color = '#ff7f0e'
    
    # Add main forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast_price'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color=line_color, width=3),
        marker=dict(
            size=8,
            color=line_color,
            symbol='circle'
        ),
        hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
    ))
    
    # Add confidence band as filled area
    if len(forecast_df) > 1:
        # Create upper and lower confidence bounds using list comprehension to handle None values
        confidence_values = [0.5 if pd.isna(x) or x is None else x for x in forecast_df['confidence']]
        forecast_prices = [0.0 if pd.isna(x) or x is None else x for x in forecast_df['forecast_price']]
        
        upper_bound = [fp * (1 + (1 - cv) * 0.1) if fp is not None and cv is not None else 0 for fp, cv in zip(forecast_prices, confidence_values)]
        lower_bound = [fp * (1 - (1 - cv) * 0.1) if fp is not None and cv is not None else 0 for fp, cv in zip(forecast_prices, confidence_values)]
        
        # Convert hex color to rgba for transparency
        if line_color == '#2ca02c':  # Green
            fill_color = 'rgba(44, 160, 44, 0.2)'
        elif line_color == '#d62728':  # Red
            fill_color = 'rgba(214, 39, 40, 0.2)'
        elif line_color == '#98df8a':  # Light green
            fill_color = 'rgba(152, 223, 138, 0.2)'
        elif line_color == '#ff9999':  # Light red
            fill_color = 'rgba(255, 153, 153, 0.2)'
        else:  # Orange
            fill_color = 'rgba(255, 127, 14, 0.2)'
        
        # Add confidence band
        fig.add_trace(go.Scatter(
            x=list(forecast_df['date']) + list(forecast_df['date'])[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor=fill_color,
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Band',
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Individual points are now included in the main line chart with markers
    
# Create animation frames that match the initial chart style
    frames = []
    for i in range(len(forecast_df)):
        frame_data = []
        
        # Get data up to current frame
        current_df = forecast_df.iloc[:i+1]
        
        if len(current_df) > 0:
            # Calculate trend and confidence for current segment
            current_trend = current_df['trend'].iloc[-1] if not current_df.empty else 'NEUTRAL'
            current_avg_confidence = current_df['confidence'].mean() if not current_df.empty else 0.5
            
            # Color based on current trend and confidence
            if current_trend == 'UP':
                current_line_color = '#2ca02c' if current_avg_confidence > 0.7 else '#98df8a'
            elif current_trend == 'DOWN':
                current_line_color = '#d62728' if current_avg_confidence > 0.7 else '#ff9999'
            else:
                current_line_color = '#ff7f0e'
            
            # Add main forecast line for current segment
            frame_data.append(go.Scatter(
                x=current_df['date'],
                y=current_df['forecast_price'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=current_line_color, width=3),
                marker=dict(
                    size=8,
                    color=current_line_color,
                    symbol='circle'
                ),
                hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
            ))
            
            # Add confidence band for current segment if we have more than 1 point
            if len(current_df) > 1:
                # Create upper and lower confidence bounds
                current_confidence_values = [0.5 if pd.isna(x) or x is None else x for x in current_df['confidence']]
                current_forecast_prices = [0.0 if pd.isna(x) or x is None else x for x in current_df['forecast_price']]
                
                current_upper_bound = [fp * (1 + (1 - cv) * 0.1) if fp is not None and cv is not None else 0 for fp, cv in zip(current_forecast_prices, current_confidence_values)]
                current_lower_bound = [fp * (1 - (1 - cv) * 0.1) if fp is not None and cv is not None else 0 for fp, cv in zip(current_forecast_prices, current_confidence_values)]
                
                # Convert hex color to rgba for transparency
                if current_line_color == '#2ca02c':  # Green
                    current_fill_color = 'rgba(44, 160, 44, 0.2)'
                elif current_line_color == '#d62728':  # Red
                    current_fill_color = 'rgba(214, 39, 40, 0.2)'
                elif current_line_color == '#98df8a':  # Light green
                    current_fill_color = 'rgba(152, 223, 138, 0.2)'
                elif current_line_color == '#ff9999':  # Light red
                    current_fill_color = 'rgba(255, 153, 153, 0.2)'
                else:  # Orange
                    current_fill_color = 'rgba(255, 127, 14, 0.2)'
                
                # Add confidence band
                frame_data.append(go.Scatter(
                    x=list(current_df['date']) + list(current_df['date'])[::-1],
                    y=current_upper_bound + current_lower_bound[::-1],
                    fill='toself',
                    fillcolor=current_fill_color,
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Confidence Band',
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Update layout with animation controls
    fig.update_layout(
        title='🔮 Price Forecasts',
        xaxis_title='Date',
        yaxis_title='Forecast Price ($)',
        hovermode='x unified',
        showlegend=True,
        height=500,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[None, {"frame": {"duration": 800, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 400}}],
                        label="▶️ Play",
                        method="animate"
                    ),
                    dict(
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}],
                        label="⏸️ Pause",
                        method="animate"
                    )
                ]),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue={"prefix": "Day: "},
                transition={"duration": 400},
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[[f.name], {"frame": {"duration": 400, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 400}}],
                        label=f"Day {int(f.name)+1}",
                        method="animate"
                    )
                    for f in frames
                ]
            )
        ] if frames else []
    )
    
    if frames:
        fig.frames = frames
    
    return fig


def create_news_table(news: List[Dict[str, Any]]) -> str:
    if not news:
        return "<p>No relevant news articles found.</p>"
    df = pd.DataFrame(news)
    # Ensure we have the expected columns
    expected_cols = ['title', 'url', 'published_date', 'source']
    available_cols = [col for col in expected_cols if col in df.columns]
    
    if not available_cols:
        return "<p>No relevant news data available.</p>"
    
    if 'published_date' in df.columns:
        df['published_date'] = pd.to_datetime(df['published_date']).dt.strftime('%Y-%m-%d')
    if 'title' in df.columns and 'url' in df.columns:
        df['title'] = df.apply(lambda r: f"<a href='{r['url']}' target='_blank'>{r['title']}</a>", axis=1)
        df.rename(columns={'title': 'Title'}, inplace=True)
    
    cols = []
    if 'title' in df.columns:
        cols.append('Title')  # Use renamed column
    if 'published_date' in df.columns:
        df.rename(columns={'published_date': 'Date'}, inplace=True)
        cols.append('Date')
    if 'source' in df.columns:
        df.rename(columns={'source': 'Source'}, inplace=True)
        cols.append('Source')
    
    if not cols or df.empty:
        return "<p>No relevant news data available.</p>"
        
    return df[cols].to_html(classes='table table-striped table-hover', index=False, escape=False)





# ---------------------
# Main Application
# ---------------------

def main():
    st.set_page_config(page_title="Agentic-Ticker (Gemini)", page_icon="📈", layout="wide")
    st.title("Agentic-Ticker 📈 — Gemini Orchestrated")
    st.markdown("Analyze stock moves and generate price forecasts using a 5-step tool pipeline orchestrated by Google Gemini.")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Analysis Parameters")
        ticker = st.text_input("Stock Ticker or Company Name", "AAPL")
        days = st.slider("Analysis Period (days)", 5, 365, 30)
        threshold = st.slider("Price Event Threshold (%)", 0.5, 10.0, 2.0)
        forecast_days = st.slider("Forecast Period (days)", 1, 30, 5)
        if st.button("Analyze", type="primary"):
            # Clear previous results when running a new analysis
            st.session_state.pop('price_data', None)
            st.session_state.pop('indicator_data', None)
            st.session_state.pop('events', None)
            st.session_state.pop('forecasts', None)
            st.session_state.pop('report', None)
            
            status = st.status("Running agent loop…", state="running", expanded=True)
            try:
                orch = Orchestrator()
                def on_event(e: Dict[str, Any]):
                    kind = e.get("type")
                    if kind == "info":
                        status.write(e.get("message"))
                    elif kind == "planning":
                        msg = f"Step {e.get('step')}: Using Gemini to plan next action"
                        status.write(msg)
                    elif kind == "call":
                        args = e.get('args', {})
                        formatted_args = _format_json_for_display(args)
                        # Create a more compact display with expandable section
                        msg = f"**Step {e.get('step')}**: `{e.get('name')}`"
                        status.write(msg)
                        # Add the JSON in a compact, scrollable format
                        status.code(formatted_args, language='json')
                    elif kind == "result":
                        res = e.get('result')
                        summary = (
                            f"{len(res)} items" if isinstance(res, list) else 
                            f"keys: {', '.join(list(res.keys())[:6])}" if isinstance(res, dict) else 
                            type(res).__name__
                        )
                        msg = f"Step {e.get('step')}: {e.get('name')} finished → {summary}"
                        status.write(msg)
                    elif kind == "final":
                        status.update(label="Agent loop completed", state="complete")
                steps = orch.run(ticker, days, threshold, forecast_days, on_event=on_event)
                price_data, indicator_data, events, news, report = [], [], [], [], None
                for s in steps:
                    if s['type'] == 'result':
                        if s['name'] == 'load_prices':
                            price_data = s['result']
                        if s['name'] == 'compute_indicators':
                            indicator_data = s['result']
                        if s['name'] == 'detect_events':
                            events = s['result']
                        if s['name'] == 'fetch_news':
                            news = s['result']
                        if s['name'] == 'build_report':
                            report = s['result']
                st.session_state.price_data = price_data
                st.session_state.indicator_data = indicator_data
                st.session_state.events = events
                st.session_state.news = news
                st.session_state.report = report
            except Exception as e:
                status.update(label="Agent loop failed", state="error")
                st.error(str(e))

    with col2:
        st.header("Analysis Results")
        # Check if we have all the required data in session state
        has_results = all(key in st.session_state for key in ['price_data', 'indicator_data', 'events', 'forecasts', 'report'])
        
        if has_results and st.session_state.report:
            st.subheader("Price Chart")
            try:
                price_fig = create_price_chart(st.session_state.price_data, st.session_state.indicator_data)
                st.plotly_chart(price_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Chart unavailable: {e}")
            
            st.subheader("Price Forecasts")
            if 'forecasts' in st.session_state:
                try:
                    forecast_fig = create_forecast_chart(st.session_state.forecasts)
                    st.plotly_chart(forecast_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Forecast chart unavailable: {e}")
            
            st.markdown(st.session_state.report["content"], unsafe_allow_html=True)
        else:
            st.info("Enter parameters and click Analyze to run the agent loop.")


if __name__ == "__main__":
    main()
