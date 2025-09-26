import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Import from src modules
from src.orchestrator import Orchestrator
from src.ui_components import create_price_chart, create_forecast_chart
from src.json_helpers import _format_json_for_display

load_dotenv(find_dotenv(), override=False)


# ---------------------
# Helper Functions
# ---------------------

def clear_all_results():
    """Clear all analysis results from session state"""
    keys_to_clear = [
        'price_data', 'indicator_data', 'events', 'forecasts', 
        'report', 'company_info', 'crypto_info', 'web_search_results', 'validated_ticker', 'analysis_running', 'analysis_completed', 'analysis_params'
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


# ---------------------
# Main Application
# ---------------------

def main():
    st.set_page_config(page_title="Agentic-Ticker (Gemini)", page_icon="📈", layout="wide")
    
    # Centered header section
    col_header_left, col_header_center, col_header_right = st.columns([1, 3, 1])
    with col_header_center:
        st.title("Agentic-Ticker 📈 — Gemini Orchestrated", anchor="header", width='content')
        st.markdown("A tool to mock up an agentic AI that analyzes stock and cryptocurrency tickers using Google Gemini and various data sources.")
        
        # Add disclaimer
        st.warning("⚠️ **Disclaimer**: This tool is designed to demonstrate how Agentic AI works and is **not intended for actual financial analysis or investment decisions**. The results are for educational purposes only.")

    # Centered parameters section - same width as header
    col_params_left, col_params_center, col_params_right = st.columns([1, 3, 1])

    with col_params_center:
        st.header("Analysis Parameters")
        ticker = st.text_input("Stock/Crypto Ticker or a description of the asset", placeholder="e.g. AAPL, BTC, or 'Apple Inc. stock' or 'Largest cryptocurrency by market cap'")
        days = st.slider("Analysis Period (days)", 5, 365, 30)
        threshold = st.slider("Price Event Threshold (%)", 0.5, 10.0, 2.0)
        forecast_days = st.slider("Forecast Period (days)", 1, 30, 5)
        
        col_analyze, col_reset = st.columns([3, 1])
        
        with col_analyze:
            if st.button("🚀 Analyze", type="primary"):
                # Clear all previous results when running a new analysis
                clear_all_results()
                # Store parameters and set analysis running flag
                st.session_state.analysis_params = {
                    'ticker': ticker,
                    'days': days,
                    'threshold': threshold,
                    'forecast_days': forecast_days
                }
                st.session_state.analysis_running = True
                st.rerun()
        
        with col_reset:
            if st.button("Reset"):
                # Clear all results and reset to initial state
                clear_all_results()
                st.rerun()

    # Bottom section for logs and report
    col_logs, col_report = st.columns(2)

    with col_logs:
        st.header("Analysis Logs")
        
        # Create a placeholder for logs
        logs_placeholder = st.empty()
        
        # Run analysis if triggered
        if st.session_state.get('analysis_running', False):
            params = st.session_state.analysis_params
            
            with logs_placeholder.container():
                status = st.status("Running agent loop…", state="running", expanded=True)
                try:
                    orch = Orchestrator()
                    def on_event(e):
                        kind = e.get("type")
                        if kind == "info":
                            status.write(e.get("message"))
                        elif kind == "planning":
                            # Skip planning display - will be combined with call event
                            pass
                        elif kind == "classification":
                            asset_type = e.get("asset_type", "unknown")
                            input_text = e.get("input", "")
                            reasoning = e.get("reasoning", "")
                            # Display classification with appropriate emoji and color
                            if asset_type == "stock":
                                emoji = "📈"
                                color = "blue"
                            elif asset_type == "crypto":
                                emoji = "₿"
                                color = "orange"
                            else:
                                emoji = "❓"
                                color = "gray"
                            msg = f":{color}[**Asset Classification:** {emoji} Classified '{input_text}' as **{asset_type.upper()}**]"
                            status.write(msg)
                            if reasoning:
                                status.write(f":{color}[Reasoning: {reasoning}]")
                        elif kind == "call":
                            args = e.get('args', {})
                            formatted_args = _format_json_for_display(args)
                            func_name = e.get('name')
                            
                            # Create agentic thinking message based on function name
                            if func_name == "ddgs_search":
                                query = args.get('query', 'unknown asset')
                                msg = f":red[**Step {e.get('step')}:**] Web search needed: I need to gather information about '{query}' to better understand this asset. so lets call `{func_name}`"
                            elif func_name == "validate_ticker":
                                input_text = args.get('input_text', 'unknown asset')
                                msg = f":red[**Step {e.get('step')}:**] Validating ticker: I need to validate and extract the correct ticker symbol from '{input_text}'. so lets call `{func_name}`"
                            
                            elif func_name == "get_company_info":
                                msg = f":red[**Step {e.get('step')}:**] Getting company details: Now that I have the ticker, I'll gather comprehensive company information. so lets call `{func_name}`"
                            elif func_name == "get_crypto_info":
                                msg = f":red[**Step {e.get('step')}:**] Getting crypto details: Now that I have the crypto symbol, I'll gather cryptocurrency-specific information. so lets call `{func_name}`"
                            elif func_name == "load_prices":
                                msg = f":red[**Step {e.get('step')}:**] Loading price data: I need historical price data to perform technical analysis. so lets call `{func_name}`"
                            elif func_name == "load_crypto_prices":
                                msg = f":red[**Step {e.get('step')}:**] Loading crypto price data: I need historical cryptocurrency price data to perform technical analysis. so lets call `{func_name}`"
                            elif func_name == "compute_indicators":
                                msg = f":red[**Step {e.get('step')}:**] Computing technical indicators: I'll calculate RSI, MACD, and Bollinger Bands for analysis. so lets call `{func_name}`"
                            elif func_name == "detect_events":
                                threshold = args.get('threshold', 2.0)
                                msg = f":red[**Step {e.get('step')}:**] Detecting significant events: I'll identify important price movements based on the {threshold}% threshold. so lets call `{func_name}`"
                            elif func_name == "forecast_prices":
                                days = args.get('days', 7)
                                msg = f":red[**Step {e.get('step')}:**] Generating price forecasts: I'll use ML models to predict price movements for the next {days} days. so lets call `{func_name}`"
                            elif func_name == "build_report":
                                msg = f":red[**Step {e.get('step')}:**] Building comprehensive report: I'll synthesize all gathered information into a final analysis report. so lets call `{func_name}`"
                            else:
                                msg = f":red[**Step {e.get('step')}:**] I need to call `{func_name}` to continue my analysis"
                            
                            status.write(msg)
                            status.code(formatted_args, language='json')
                        elif kind == "result":
                            res = e.get('result')
                            name = e.get('name')
                                                                                
                            # Show data directly without expander
                            if isinstance(res, (list, dict)) and len(res) > 0:
                                if isinstance(res, list):
                                    # Show first 10 items
                                    display_data = res[:10]
                                    formatted_data = _format_json_for_display(display_data)
                                    status.code(formatted_data, language='json')
                                else:
                                    formatted_data = _format_json_for_display(res)
                                    status.code(formatted_data, language='json')
                        elif kind == "final":
                            status.update(label="Agent loop completed", state="complete")
                    
                    steps = orch.run(params['ticker'], params['days'], params['threshold'], params['forecast_days'], on_event=on_event)
                    price_data, indicator_data, events, forecasts, report, company_info, crypto_info, web_search_results, validated_ticker = [], [], [], [], None, None, None, None, None
                    for s in steps:
                        if s['type'] == 'result':
                            if s['name'] == 'load_prices':
                                price_data = s['result']
                            if s['name'] == 'load_crypto_prices':
                                price_data = s['result']
                            if s['name'] == 'compute_indicators':
                                indicator_data = s['result']
                            if s['name'] == 'detect_events':
                                events = s['result']
                            if s['name'] == 'forecast_prices':
                                forecasts = s['result']
                            if s['name'] == 'build_report':
                                report = s['result']
                            if s['name'] == 'get_company_info':
                                company_info = s['result']
                            if s['name'] == 'get_crypto_info':
                                crypto_info = s['result']
                            if s['name'] == 'ddgs_search':
                                web_search_results = s['result']
                            if s['name'] in ['validate_ticker']:
                                validated_ticker = s['result']
                    
                    st.session_state.price_data = price_data
                    st.session_state.indicator_data = indicator_data
                    st.session_state.events = events
                    st.session_state.forecasts = forecasts
                    st.session_state.report = report
                    st.session_state.company_info = company_info
                    st.session_state.crypto_info = crypto_info
                    st.session_state.web_search_results = web_search_results
                    st.session_state.validated_ticker = validated_ticker or params['ticker']
                    # Clear the analysis running flag and set completed flag
                    st.session_state.analysis_running = False
                    st.session_state.analysis_completed = True
                    # Don't rerun here - let both logs and report show side by side
                except Exception as e:
                    status.update(label="Agent loop failed", state="error")
                    st.error(str(e))
                    # Clear the analysis running flag even on error
                    st.session_state.analysis_running = False
                    # Don't rerun on error either - let both logs and report show side by side
        elif st.session_state.get('analysis_completed', False):
            # Show completed logs
            with logs_placeholder.container():
                st.success("✅ Analysis completed successfully!")
                st.info("Analysis logs are shown above. Results are displayed in the report section on the right.")
        else:
            logs_placeholder.info("Run an analysis to see logs here.")

    with col_report:
        st.header("Analysis Report")
        
        # Check if we have results to display
        has_results = all(key in st.session_state for key in ['price_data', 'indicator_data', 'events', 'forecasts', 'report'])
        
        if has_results and st.session_state.report:
            # Display unified header with company/crypto name and validated ticker
            company_info = st.session_state.get('company_info', {})
            crypto_info = st.session_state.get('crypto_info', {})
            validated_ticker = st.session_state.get('validated_ticker', st.session_state.get('analysis_params', {}).get('ticker', 'Unknown'))
            
            # Determine asset name based on available info
            if crypto_info and crypto_info.get('name'):
                asset_name = crypto_info.get('name')
            elif company_info and company_info.get('company_name'):
                asset_name = company_info.get('company_name')
            else:
                asset_name = validated_ticker
                
            st.header(f"📊 Results for {asset_name} ({validated_ticker})")
            st.markdown("---")  # Separator line
            
            st.subheader("📈 Price Chart")
            try:
                price_fig = create_price_chart(st.session_state.price_data, st.session_state.indicator_data)
                st.plotly_chart(price_fig)
            except Exception as e:
                st.warning(f"Chart unavailable: {e}")
            
            st.subheader("🔮 Price Forecasts")
            if 'forecasts' in st.session_state:
                try:
                    forecast_fig = create_forecast_chart(st.session_state.forecasts)
                    st.plotly_chart(forecast_fig)
                except Exception as e:
                    st.warning(f"Forecast chart unavailable: {e}")
            
            if 'report' in st.session_state and st.session_state.report:
                report_content = st.session_state.report.get('content', 'No report content available.')
                st.markdown(report_content)
            else:
                st.info("No report available.")
        else:
            st.info("Run an analysis to see results here.")


if __name__ == "__main__":
    main()