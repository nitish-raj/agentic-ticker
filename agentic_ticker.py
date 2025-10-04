#!/usr/bin/env python3
"""
Agentic Ticker - Streamlit Application
A working Streamlit application that handles import issues gracefully.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import data formatter for display formatting
try:
    from src.data_formatter import (
        format_price_data, 
        format_indicator_data, 
        format_events_data, 
        format_forecasts_data
    )
    
except ImportError as e:
    # Fallback if formatter not available
    
    def format_price_data(data):
        return pd.DataFrame(data)
    def format_indicator_data(data):
        return pd.DataFrame(data)
    def format_events_data(data):
        return pd.DataFrame(data)
    def format_forecasts_data(data):
        return pd.DataFrame(data)

# Import session state manager for thread-safe operations
try:
    from src.session_state_manager import (
        session_manager, 
        set_session_state, 
        get_session_state, 
        update_session_state, 
        delete_session_state, 
        clear_session_state
    )
except ImportError:
    # Fallback if import fails - will be defined after streamlit import
    session_manager = None

# Load configuration with security features
try:
    import importlib.util
    
    config_security_module = None
    
    # Try to load secure configuration first
    config_security_path = project_root / "src" / "config_security.py"
    if config_security_path.exists():
        spec = importlib.util.spec_from_file_location("config_security", config_security_path)
        if spec is None:
            raise ImportError(f"Could not load config security module from {config_security_path}")
        config_security_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError("Config security module loader not available")
        spec.loader.exec_module(config_security_module)
        config = config_security_module.load_config_with_env_fallback()
        print("✅ Configuration loaded with environment variable support")
    else:
        # Fallback to regular configuration
        config_path = project_root / "src" / "config.py"
        spec = importlib.util.spec_from_file_location("app_config", config_path)
        if spec is None:
            raise ImportError(f"Could not load config module from {config_path}")
        config_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError("Config module loader not available")
        spec.loader.exec_module(config_module)
        config = config_module.get_config()
    
    # Validate security configuration
    if config_security_module and hasattr(config_security_module, 'validate_security_config'):
        security_results = config_security_module.validate_security_config()
        if security_results["warnings"]:
            for warning in security_results["warnings"]:
                print(f"⚠️  Security warning: {warning}")
        if security_results["recommendations"]:
            print("🔒 Security recommendations:")
            for rec in security_results["recommendations"]:
                print(f"   - {rec}")
    
except Exception as e:
    print(f"❌ Failed to load configuration: {e}")
    sys.exit(1)

# Enable compatibility mode (temporarily, will be moved to config.yaml later)
import os
os.environ['COMPATIBILITY_ENABLED'] = 'true'

# Now import other modules that might depend on environment variables
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Define fallback session state functions if import failed
if session_manager is None:
    def set_session_state(key, value): 
        st.session_state[key] = value
    def get_session_state(key, default=None): 
        return st.session_state.get(key, default)
    def update_session_state(updates): 
        for k, v in updates.items(): 
            st.session_state[k] = v
    def delete_session_state(keys): 
        for key in keys: 
            if key in st.session_state: 
                del st.session_state[key]
    def clear_session_state(): 
        for key in list(st.session_state.keys()): 
            del st.session_state[key]

# ---------------------
# Compatibility Layer
# ---------------------

# Use real API calls instead of mock data
MOCK_MODE = False
Orchestrator = None  # Initialize to None

# Import real orchestrator when not in mock mode
if not MOCK_MODE:
    try:
        # Check if Gemini API key is available from config
        gemini_api_key = None
        if config and config.gemini and config.gemini.api_key:
            gemini_api_key = config.gemini.api_key
            print("✅ Gemini API key found in configuration")
        else:
            print("❌ Gemini API key not found in configuration")
            print("💡 Tip: Edit config.yaml and add your Gemini API key")
        
        if gemini_api_key:
            print("✅ Gemini API key is available, proceeding with real orchestrator")
            from src.orchestrator import Orchestrator
            print("✅ Real Orchestrator imported successfully")
        else:
            print("❌ No Gemini API key found - falling back to mock mode")
            print("💡 Tip: Edit config.yaml and add your Gemini API key")
            MOCK_MODE = True
            
    except ImportError as e:
        print(f"❌ Failed to import real Orchestrator: {e}")
        print("⚠️  Falling back to mock mode")
        MOCK_MODE = True
    except Exception as e:
        print(f"❌ Error during orchestrator setup: {e}")
        print("⚠️  Falling back to mock mode")
        MOCK_MODE = True

# ---------------------
# Compatibility Layer
# ---------------------

# Use real API calls instead of mock data
MOCK_MODE = False

def create_mock_orchestrator():
    """Create a mock orchestrator that simulates the AI analysis"""
    class MockOrchestrator:
        def run(self, ticker_input, days, threshold, forecast_days, on_event=None):
            """Simulate the agentic AI analysis process"""
            
            # Simulate the step-by-step process
            steps = [
                ("Initializing Analysis", "Setting up analysis parameters"),
                ("Web Search", f"Searching for information about {ticker_input}"),
                ("Asset Validation", f"Validating ticker/asset: {ticker_input}"),
                ("Data Collection", "Collecting historical price data"),
                ("Technical Analysis", "Computing technical indicators (RSI, MACD, Bollinger Bands)"),
                ("Event Detection", f"Detecting significant price events (threshold: {threshold}%)"),
                ("Price Forecasting", f"Generating {forecast_days}-day price forecast"),
                ("Report Generation", "Compiling comprehensive analysis report"),
                ("Analysis Complete", "Finalizing results")
            ]
            
            for i, (step_name, description) in enumerate(steps):
                if on_event:
                    on_event({
                        'type': 'info' if i == 0 else 'call' if i < len(steps) - 1 else 'final',
                        'name': step_name,
                        'message': description,
                        'step': i + 1
                    })
                
                # Simulate processing time
                import time
                time.sleep(0.3)
            
            return steps
    
    return MockOrchestrator()

def create_mock_data():
    """Create realistic mock data for demonstration"""
    import numpy as np
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate mock price data with realistic patterns
    np.random.seed(42)  # For reproducible results
    base_price = 150.0
    trend = np.linspace(0, 10, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    prices = base_price + trend + np.cumsum(noise)
    
    # Ensure positive prices
    prices = np.maximum(prices, base_price * 0.8)
    
    price_data = []
    for i, date in enumerate(dates):
        price_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': prices[i] * 0.99,
            'high': prices[i] * 1.02,
            'low': prices[i] * 0.98,
            'close': prices[i],
            'volume': np.random.randint(1000000, 5000000)
        })
    
    # Generate technical indicators
    indicator_data = []
    for i, date in enumerate(dates):
        indicator_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'rsi': 50 + np.random.normal(0, 15),
            'macd': np.random.normal(0, 1),
            'bollinger_upper': prices[i] * 1.05,
            'bollinger_lower': prices[i] * 0.95,
            'bollinger_position': 'Middle' if i % 3 == 0 else 'Upper' if i % 3 == 1 else 'Lower'
        })
    
    # Generate events
    events = []
    event_dates = [dates[i] for i in [5, 12, 18, 25]]
    event_descriptions = [
        "Strong earnings report exceeded expectations",
        "Market volatility due to economic news",
        "Sector rotation into technology stocks",
        "Analyst upgrade with increased price target"
    ]
    
    for i, (date, description) in enumerate(zip(event_dates, event_descriptions)):
        events.append({
            'date': date.strftime('%Y-%m-%d'),
            'description': description,
            'magnitude': round(np.random.uniform(1.5, 4.0), 1),
            'type': 'positive' if i % 2 == 0 else 'negative'
        })
    
    # Generate forecasts
    forecast_dates = pd.date_range(start=end_date + timedelta(days=1), periods=5, freq='D')
    current_price = prices[-1]
    forecast_prices = current_price + np.cumsum(np.random.normal(0.5, 1, 5))
    
    forecasts = []
    for i, (date, price) in enumerate(zip(forecast_dates, forecast_prices)):
        forecasts.append({
            'date': date.strftime('%Y-%m-%d'),
            'predicted_price': round(price, 2),
            'confidence': f"{85 + np.random.randint(-10, 10)}%",
            'trend': 'Bullish' if price > current_price else 'Bearish' if price < current_price else 'Neutral'
        })
    
    return price_data, indicator_data, events, forecasts

def check_api_key_availability():
    """Check if Gemini API key is available from config.yaml"""
    # Check configuration system
    try:
        import importlib.util
        config_path = project_root / "src" / "config.py"
        if config_path.exists():
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec is not None:
                config_module = importlib.util.module_from_spec(spec)
                if spec.loader is not None:
                    spec.loader.exec_module(config_module)
                    config = config_module.get_config()
                    if config.gemini.api_key:
                        return True, "configuration"
    except Exception:
        pass
    
    return False, None

def create_mock_report(ticker, days, threshold, forecast_days):
    """Create a comprehensive mock analysis report"""
    
    return {
        'asset_info': {
            'type': 'stock',
            'company_name': f'{ticker} Corporation',
            'sector': 'Technology',
            'market_cap': '$2.8T',
            'description': f'{ticker} is a leading technology company known for innovation and strong financial performance.'
        },
        'price_analysis': {
            'current_price': '$175.50',
            'price_change_30d': '+5.2%',
            'volatility': 'Medium (2.1%)',
            'trading_volume': '45.2M shares',
            'market_trend': 'Slightly Bullish'
        },
        'technical_indicators': {
            'rsi': '65.3 (Neutral)',
            'macd_signal': 'Bullish Crossover',
            'bollinger_position': 'Upper Band (Strong)',
            'moving_average_50d': '$168.20',
            'moving_average_200d': '$155.80',
            'support_level': '$165.00',
            'resistance_level': '$180.00'
        },
        'events': [
            {'date': '2024-01-15', 'description': 'Strong earnings report exceeded expectations', 'magnitude': 3.2, 'impact': 'positive'},
            {'date': '2024-01-20', 'description': 'Market volatility due to economic news', 'magnitude': -1.8, 'impact': 'negative'},
            {'date': '2024-01-25', 'description': 'Sector rotation into technology stocks', 'magnitude': 2.1, 'impact': 'positive'}
        ],
        'forecast': {
            'predicted_price_5d': '$178.25',
            'confidence': '72%',
            'trend': 'Slightly Bullish',
            'expected_range': '$172.00 - $184.50',
            'key_factors': ['Strong technical indicators', 'Positive market sentiment', 'Sector momentum']
        },
        'web_search_results': [
            {
                'title': f'{ticker} Stock Analysis - Strong Buy Rating',
                'href': '#',
                'content': f'Recent analysis shows {ticker} maintaining strong fundamentals with positive outlook for Q1 earnings...',
                'source': 'Financial News',
                'date': '2024-01-28'
            },
            {
                'title': f'{ticker} Technology Innovation Drives Growth',
                'href': '#',
                'content': f'{ticker} continues to invest heavily in R&D, positioning itself for long-term growth in emerging markets...',
                'source': 'Tech Weekly',
                'date': '2024-01-27'
            }
        ],
        'analysis_summary': {
            'overall_rating': 'BUY',
            'risk_level': 'Medium',
            'investment_horizon': 'Medium-term (3-6 months)',
            'key_strengths': ['Strong technical position', 'Positive sector trends', 'Solid fundamentals'],
            'key_risks': ['Market volatility', 'Economic uncertainty', 'Sector rotation'],
            'recommendation': 'Consider accumulating on dips with stop-loss at $165'
        }
    }

# ---------------------
# Streamlit App
# ---------------------

def main():
    # Page configuration
    st.set_page_config(
        page_title="Agentic Ticker 🤖",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
            font-weight: 500;
        }
        .status-demo {
            background-color: #e3f2fd;
            color: #1976d2;
            border: 1px solid #bbdefb;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header section
    st.markdown('<div class="main-header">🤖 Agentic Ticker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Stock & Cryptocurrency Analysis powered by Agentic AI</div>', unsafe_allow_html=True)
    
    # Status indicator
    is_mock_mode = MOCK_MODE or get_session_state('mock_mode_override', False)
    if is_mock_mode:
        st.markdown('<div class="status-badge status-demo">📊 Demo Mode - Educational Purpose Only</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge" style="background-color: #e8f5e8; color: #2e7d32; border: 1px solid #c8e6c9;">🚀 Live Mode - Real API Data</div>', unsafe_allow_html=True)
    
    # Disclaimer
    with st.expander("⚠️ Important Disclaimer", expanded=False):
        st.warning("""
        **Educational Purpose Only**: This tool demonstrates Agentic AI principles and is **not intended for actual financial decisions**. 
        
        **What it does**:
        - Simulates AI-powered financial analysis
        - Demonstrates agentic workflow orchestration
        - Shows how AI can chain multiple analysis steps
        - Provides educational insights into AI decision-making
        
        **What it doesn't do**:
        - Provide real financial advice
        - Make actual predictions
        - Replace professional financial analysis
        - Guarantee investment outcomes
        """)
    
    # How it works section
    with st.expander("🧠 How Agentic AI Works", expanded=False):
        st.markdown("""
        This system demonstrates **Agentic AI** where the AI (Google Gemini) autonomously:
        
        1. **Analyzes** your input to understand what asset you want to analyze
        2. **Decides** which functions to call and in what order
        3. **Orchestrates** a sequence of analysis steps:
           - Web search for context
           - Asset validation and classification
           - Historical data collection
           - Technical indicator calculation
           - Event detection
           - Price forecasting
           - Report generation
        4. **Adapts** its strategy based on available data and results
        5. **Explains** its reasoning for each decision
        
        The AI doesn't just follow a script - it **thinks** through the analysis process!
        """)
    
    # Centered parameters section
    col_params_left, col_params_center, col_params_right = st.columns([1, 3, 1])

    with col_params_center:
        st.header("🎯 Analysis Parameters")
        
        ticker = st.text_input(
            "Stock/Crypto Ticker or Asset Description",
            placeholder="e.g. AAPL, BTC, 'Apple Inc. stock', 'Largest cryptocurrency'",
            help="Enter a ticker symbol, company name, or description of the asset you want to analyze"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            days = st.slider("Analysis Period (days)", 5, 365, 30, help="How many days of historical data to analyze")
            threshold = st.slider("Price Event Threshold (%)", 0.5, 10.0, 2.0, 0.1, help="Minimum price change to consider as significant event")
        with col2:
            forecast_days = st.slider("Forecast Period (days)", 1, 30, 5, help="How many days ahead to forecast")
        
        col_analyze, col_reset = st.columns([3, 1])
        
        with col_analyze:
            analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
        
        with col_reset:
            reset_button = st.button("🔄 Reset", use_container_width=True)
        
        if reset_button:
            clear_session_state()
            st.rerun()

    # Analysis execution section
    if analyze_button and ticker:
        # Store parameters and set analysis running flag atomically
        update_session_state({
            'analysis_params': {
                'ticker': ticker,
                'days': days,
                'threshold': threshold,
                'forecast_days': forecast_days
            },
            'analysis_running': True
        })
        st.rerun()

    if get_session_state('analysis_running', False):
        params = get_session_state('analysis_params')
        
        # Create progress containers
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            st.header("🔍 Analysis Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            step_container = st.empty()
        
        # Initialize orchestrator with proper error handling
        orchestrator = None
        try:
            if MOCK_MODE or Orchestrator is None:
                orchestrator = create_mock_orchestrator()
                print("🎯 Using mock orchestrator for demo mode")
            else:
                # Check API key availability
                api_key_available, source = check_api_key_availability()
                
                if api_key_available:
                    orchestrator = Orchestrator()
                    print("🚀 Real orchestrator initialized successfully (API key available)")
                else:
                    print("⚠️  API key not available, falling back to mock mode")
                    print("💡 Tip: Add GEMINI_API_KEY to your config.yaml file")
                    orchestrator = create_mock_orchestrator()
                    # Update the global MOCK_MODE for UI consistency
                    set_session_state('mock_mode_override', True)
        except Exception as init_error:
            # Sanitize error message to prevent API key exposure
            try:
                from src.sanitization import sanitize_error_message
                sanitized_error = sanitize_error_message(str(init_error))
            except ImportError:
                sanitized_error = str(init_error)
            print(f"❌ Failed to initialize orchestrator: {sanitized_error}")
            print("⚠️  Falling back to mock mode")
            orchestrator = create_mock_orchestrator()
            # Update the global MOCK_MODE for UI consistency
            set_session_state('mock_mode_override', True)
        
        # Event callback for progress updates
        def on_event(event):
            step = event.get('step', 0)
            progress = min(step * 11, 100)  # 9 steps = ~11% each
            progress_bar.progress(progress)
            
            if event['type'] == 'info':
                message = event.get('message', 'Processing...')
                status_text.info(f"ℹ️ {message}")
            elif event['type'] == 'call':
                func_name = event.get('name', 'Unknown function')
                status_text.info(f"🔄 Calling {func_name}...")
            elif event['type'] == 'result':
                func_name = event.get('name', 'Function')
                status_text.success(f"✅ {func_name} completed")
            elif event['type'] == 'error':
                error_msg = event.get('error', 'Unknown error')
                status_text.error(f"❌ Error: {error_msg}")
            elif event['type'] == 'final':
                status_text.success("🎉 Analysis complete!")
                progress_bar.progress(100)
            elif event['type'] == 'planning':
                status_text.info(f"🧠 Planning step {step}...")
            else:
                # Handle any other event types
                status_text.info(f"📊 Processing step {step}...")
        
        try:
            # Run the analysis
            with st.spinner(f"🧠 AI is analyzing {params['ticker']}..."):
                steps = orchestrator.run(
                    ticker_input=params['ticker'],
                    days=params['days'],
                    threshold=params['threshold'],
                    forecast_days=params['forecast_days'],
                    on_event=on_event
                )
            
            # The real orchestrator already updates session state with actual data
            # No need to generate mock data when using real APIs
            if MOCK_MODE:
                # Only generate mock data if we're in mock mode
                price_data, indicator_data, events, forecasts = create_mock_data()
                report = create_mock_report(
                    params['ticker'], 
                    params['days'], 
                    params['threshold'], 
                    params['forecast_days']
                )
                
                # Store mock results in session state atomically
                update_session_state({
                    'price_data': price_data,
                    'indicator_data': indicator_data,
                    'events': events,
                    'forecasts': forecasts,
                    'report': report
                })
            # For real mode, the orchestrator's update_session_state() function 
            # already populated the session state with real data
            
            # Mark analysis as completed atomically
            update_session_state({
                'analysis_completed': True,
                'analysis_running': False
            })
            st.rerun()
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            set_session_state('analysis_running', False)
            st.rerun()

    # Results display section
    if get_session_state('analysis_completed', False):
        st.header("📊 Analysis Results")
        
        # Create tabs for different result types
        tab_summary, tab_charts, tab_data, tab_report = st.tabs(["📋 Summary", "📈 Charts", "📊 Data", "📄 Full Report"])
        
        with tab_summary:
            report = get_session_state('report')
            if report:
                
                # Key metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'price_analysis' in report:
                        st.metric("Current Price", report['price_analysis']['current_price'])
                        st.metric("30-Day Change", report['price_analysis']['price_change_30d'])
                with col2:
                    if 'technical_indicators' in report:
                        st.metric("RSI", report['technical_indicators']['rsi'])
                        st.metric("MACD Signal", report['technical_indicators']['macd_signal'])
                with col3:
                    if 'forecast' in report:
                        st.metric("5-Day Forecast", report['forecast']['predicted_price_5d'])
                        st.metric("Confidence", report['forecast']['confidence'])
                
                # Analysis summary
                if 'analysis_summary' in report:
                    summary = report['analysis_summary']
                    st.subheader("🎯 Analysis Summary")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Overall Rating:** {summary.get('overall_rating', 'N/A')}")
                        st.write(f"**Risk Level:** {summary.get('risk_level', 'N/A')}")
                        st.write(f"**Investment Horizon:** {summary.get('investment_horizon', 'N/A')}")
                    with col2:
                        st.write(f"**Trend:** {summary.get('key_strengths', ['N/A'])[0] if summary.get('key_strengths') else 'N/A'}")
                        st.write(f"**Key Factors:** {', '.join(summary.get('key_factors', ['N/A'])[:2])}")
                    
                    if 'recommendation' in summary:
                        st.info(f"💡 **Recommendation:** {summary['recommendation']}")
        
        with tab_charts:
            # Price chart
            price_data = get_session_state('price_data')
            if price_data:
                try:
                    import plotly.graph_objects as go
                    
                    df = pd.DataFrame(price_data)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df['date'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price'
                    ))
                    fig.update_layout(
                        title='Stock Price Chart',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("Plotly not available for charts. Install with: pip install plotly")
            
            # Forecast chart
            forecasts = get_session_state('forecasts')
            if forecasts:
                try:
                    import plotly.graph_objects as go
                    
                    df = pd.DataFrame(forecasts)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['date'], 
                        y=df['forecast_price'], 
                        mode='lines+markers', 
                        name='Forecast',
                        line=dict(color='blue', width=2)
                    ))
                    fig.update_layout(
                        title='Price Forecast',
                        xaxis_title='Date',
                        yaxis_title='Forecast Price ($)',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("Plotly not available for charts. Install with: pip install plotly")
        
        with tab_data:
            # Display data in expandable sections with formatting
            price_data = get_session_state('price_data')
            if price_data:
                with st.expander("📈 Price Data"):
                    formatted_price_data = format_price_data(price_data)
                    st.dataframe(formatted_price_data)
            
            indicator_data = get_session_state('indicator_data')
            if indicator_data:
                with st.expander("📊 Technical Indicators"):
                    formatted_indicator_data = format_indicator_data(indicator_data)
                    st.dataframe(formatted_indicator_data)
            
            events = get_session_state('events')
            if events:
                with st.expander("⚡ Price Events"):
                    formatted_events_data = format_events_data(events)
                    st.dataframe(formatted_events_data)
            
            forecasts = get_session_state('forecasts')
            if forecasts:
                with st.expander("🔮 Forecasts"):
                    formatted_forecasts_data = format_forecasts_data(forecasts)
                    st.dataframe(formatted_forecasts_data)
        
        with tab_report:
            # Display the comprehensive report
            report = get_session_state('report')
            if report:
                
                # Asset Information
                if 'asset_info' in report:
                    st.subheader("🏢 Asset Information")
                    info = report['asset_info']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Company:** {info.get('company_name', 'N/A')}")
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    with col2:
                        st.write(f"**Market Cap:** {info.get('market_cap', 'N/A')}")
                    if 'description' in info:
                        st.write(f"**Description:** {info['description']}")
                
                # Price Analysis
                if 'price_analysis' in report:
                    st.subheader("💰 Price Analysis")
                    price_data = report['price_analysis']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Current Price:** {price_data.get('current_price', 'N/A')}")
                        st.write(f"**30-Day Change:** {price_data.get('price_change_30d', 'N/A')}")
                    with col2:
                        st.write(f"**Volatility:** {price_data.get('volatility', 'N/A')}")
                        st.write(f"**Trading Volume:** {price_data.get('trading_volume', 'N/A')}")
                
                # Technical Indicators
                if 'technical_indicators' in report:
                    st.subheader("📊 Technical Indicators")
                    indicators = report['technical_indicators']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RSI", indicators.get('rsi', 'N/A'))
                        st.metric("MACD Signal", indicators.get('macd_signal', 'N/A'))
                    with col2:
                        st.metric("Bollinger Position", indicators.get('bollinger_position', 'N/A'))
                        st.metric("50-Day MA", indicators.get('moving_average_50d', 'N/A'))
                    with col3:
                        st.metric("200-Day MA", indicators.get('moving_average_200d', 'N/A'))
                        st.metric("Support Level", indicators.get('support_level', 'N/A'))
                
                # Events
                if 'events' in report and report['events']:
                    st.subheader("⚡ Significant Events")
                    for event in report['events']:
                        impact_emoji = "📈" if event.get('impact') == 'positive' else "📉" if event.get('impact') == 'negative' else "➡️"
                        st.write(f"{impact_emoji} **{event.get('date', 'N/A')}**: {event.get('description', 'N/A')} (Magnitude: {event.get('magnitude', 'N/A')}%)")
                
                # Forecast
                if 'forecast' in report:
                    st.subheader("🔮 Price Forecast")
                    forecast = report['forecast']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Predicted Price (5d):** {forecast.get('predicted_price_5d', 'N/A')}")
                        st.write(f"**Confidence:** {forecast.get('confidence', 'N/A')}")
                    with col2:
                        st.write(f"**Trend:** {forecast.get('trend', 'N/A')}")
                        st.write(f"**Expected Range:** {forecast.get('expected_range', 'N/A')}")
                    
                    if 'key_factors' in forecast:
                        st.write(f"**Key Factors:** {', '.join(forecast['key_factors'])}")
                
                # Web Search Results
                if 'web_search_results' in report and report['web_search_results']:
                    st.subheader("📰 Recent News & Analysis")
                    for result in report['web_search_results']:
                        st.write(f"📰 [{result.get('title', 'N/A')}]({result.get('href', '#')})")
                        st.write(f"   {result.get('content', 'N/A')[:200]}...")
                        st.caption(f"Source: {result.get('source', 'Unknown')} | {result.get('date', 'N/A')}")

    # Sidebar with system information
    with st.sidebar:
        st.header("🤖 System Information")
        
        is_mock_mode = MOCK_MODE or get_session_state('mock_mode_override', False)
        if is_mock_mode:
            st.markdown('<div class="status-badge status-demo">Demo Mode</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge" style="background-color: #e8f5e8; color: #2e7d32; border: 1px solid #c8e6c9;">Live Mode</div>', unsafe_allow_html=True)
        
        with st.expander("🔧 How This Works"):
            is_mock_mode = MOCK_MODE or get_session_state('mock_mode_override', False)
            if is_mock_mode:
                st.markdown("""
                **Agentic AI Process**:
                1. **Input Analysis** - Understand your request
                2. **Web Search** - Gather context about the asset
                3. **Asset Validation** - Confirm ticker/symbol
                4. **Data Collection** - Get historical prices
                5. **Technical Analysis** - Calculate indicators
                6. **Event Detection** - Find significant movements
                7. **Forecasting** - Predict future prices
                8. **Report Generation** - Compile results
                
                **Note**: This is a demonstration using mock data to show how agentic AI would work.
                """)
            else:
                st.markdown("""
                **Agentic AI Process**:
                1. **Input Analysis** - Understand your request
                2. **Web Search** - Gather context about the asset
                3. **Asset Validation** - Confirm ticker/symbol
                4. **Data Collection** - Get historical prices from Yahoo Finance
                5. **Technical Analysis** - Calculate indicators (RSI, MACD, Bollinger)
                6. **Event Detection** - Find significant movements
                7. **Forecasting** - Predict future prices using ML
                8. **Report Generation** - Compile results
                
                **Note**: This uses real financial APIs and live market data.
                """)
        
        with st.expander("📊 Data Source"):
            is_mock_mode = MOCK_MODE or get_session_state('mock_mode_override', False)
            if is_mock_mode:
                st.write("Current session uses realistic mock data including:")
                st.write("• 30 days of price history")
                st.write("• Technical indicators (RSI, MACD, Bollinger)")
                st.write("• Significant price events")
                st.write("• 5-day price forecasts")
                st.write("• Market news and analysis")
            else:
                st.write("Current session uses real market data from:")
                st.write("• Yahoo Finance API (stock prices)")
                st.write("• CoinGecko API (crypto prices)")
                st.write("• Google Gemini AI (analysis)")
                st.write("• DuckDuckGo (web search)")
                st.write("• Real-time technical indicators")
                st.write("• Live market news and events")
        
        with st.expander("🚀 Getting Started"):
            st.markdown("""
            **Quick Start**:
            1. Enter a ticker symbol (AAPL, BTC, etc.)
            2. Adjust analysis parameters
            3. Click "Analyze" to start
            4. View results in different tabs
            
            **Tips**:
            - Try different tickers
            - Adjust time periods
            - Compare technical indicators
            - Review the full analysis report
            """)

if __name__ == "__main__":
    main()