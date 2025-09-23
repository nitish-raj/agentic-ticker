import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Import from src modules
from src.orchestrator import Orchestrator
from src.ui_components import create_price_chart, create_forecast_chart
from src.json_helpers import _format_json_for_display
from src.services import get_company_info

load_dotenv(find_dotenv(), override=False)


# ---------------------
# Helper Functions
# ---------------------

def clear_all_results():
    """Clear all analysis results from session state"""
    keys_to_clear = [
        'price_data', 'indicator_data', 'events', 'forecasts', 
        'report', 'company_info', 'validated_ticker', 'analysis_running'
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


# ---------------------
# Main Application
# ---------------------

def main():
    st.set_page_config(page_title="Agentic-Ticker (Gemini)", page_icon="📈", layout="wide")
    st.title("Agentic-Ticker 📈 — Gemini Orchestrated")
    st.markdown("Analyze stock moves and generate price forecasts by mocking agentic behaviour using Gemini Flash")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Analysis Parameters")
        ticker = st.text_input("Stock Ticker or Company Name", "AAPL")
        days = st.slider("Analysis Period (days)", 5, 365, 30)
        threshold = st.slider("Price Event Threshold (%)", 0.5, 10.0, 2.0)
        forecast_days = st.slider("Forecast Period (days)", 1, 30, 5)
        
        col_analyze, col_reset = st.columns([3, 1])
        
        with col_analyze:
            if st.button("Analyze", type="primary"):
                # Clear all previous results when running a new analysis
                clear_all_results()
                # Set a flag to indicate analysis is running
                st.session_state.analysis_running = True
                
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
                        elif kind == "call":
                            args = e.get('args', {})
                            formatted_args = _format_json_for_display(args)
                            # Create a more compact display with expandable section
                            msg = f":red[**Step {e.get('step')}:**] Gemini calling function: `{e.get('name')}`"
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
                            # Create descriptive messages based on function name
                            name = e.get('name')
                            if name == 'validate_ticker_gemini_only':
                                description = "Validated Ticker using Gemini, which Returned the correct Symbol"
                            elif name == 'validate_ticker_with_web_search':
                                description = "Performed Web Search to find Ticker Symbol for the given company"
                            elif name == 'get_company_info':
                                description = "Retrieved Company information using YFinance, by passing the validated ticker"
                            elif name == 'load_prices':
                                description = "Loaded Historical Price Data for the specified period"
                            elif name == 'compute_indicators':
                                description = "Computed Technical Indicators (RSI, MACD, Bollinger Bands)"
                            elif name == 'detect_events':
                                description = "Detected Significant Price Events based on threshold"
                            elif name == 'forecast_prices':
                                description = "Generated Price Forecasts using ML models"
                            elif name == 'build_report':
                                description = "Built Comprehensive Analysis Report with all findings"
                            else:
                                description = f"Completed {name} operation"
                            
                            msg = f":yellow[Result: {description} → {summary}]"
                            status.write(msg)
                        elif kind == "final":
                            status.update(label="Agent loop completed", state="complete")
                    steps = orch.run(ticker, days, threshold, forecast_days, on_event=on_event)
                    price_data, indicator_data, events, forecasts, report, company_info, validated_ticker = [], [], [], [], None, None, None
                    for s in steps:
                        if s['type'] == 'result':
                            if s['name'] == 'load_prices':
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
                            if s['name'] in ['validate_ticker_gemini_only', 'validate_ticker_with_web_search']:
                                validated_ticker = s['result']
                    
                    st.session_state.price_data = price_data
                    st.session_state.indicator_data = indicator_data
                    st.session_state.events = events
                    st.session_state.forecasts = forecasts
                    st.session_state.report = report
                    st.session_state.company_info = company_info
                    st.session_state.validated_ticker = validated_ticker or ticker
                    # Clear the analysis running flag
                    st.session_state.analysis_running = False
                except Exception as e:
                    status.update(label="Agent loop failed", state="error")
                    st.error(str(e))
                    # Clear the analysis running flag even on error
                    st.session_state.analysis_running = False
        
        with col_reset:
            if st.button("Reset"):
                # Clear all results and reset to initial state
                clear_all_results()
                st.rerun()

    with col2:
        # Check if we have all the required data in session state and analysis is not running
        has_results = all(key in st.session_state for key in ['price_data', 'indicator_data', 'events', 'forecasts', 'report'])
        analysis_not_running = not st.session_state.get('analysis_running', False)
        
        if has_results and st.session_state.report and analysis_not_running:
            # Display unified header with company name and validated ticker
            company_info = st.session_state.get('company_info', {})
            validated_ticker = st.session_state.get('validated_ticker', ticker)
            company_name = company_info.get('company_name', validated_ticker) if company_info else validated_ticker
            st.header(f"📊 Results for {company_name} ({validated_ticker})")
            st.markdown("---")  # Separator line
            
            st.subheader("📈 Price Chart")
            try:
                price_fig = create_price_chart(st.session_state.price_data, st.session_state.indicator_data)
                st.plotly_chart(price_fig, width='stretch')
            except Exception as e:
                st.warning(f"Chart unavailable: {e}")
            
            st.subheader("🔮 Price Forecasts")
            if 'forecasts' in st.session_state:
                try:
                    forecast_fig = create_forecast_chart(st.session_state.forecasts)
                    st.plotly_chart(forecast_fig, width='stretch')
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