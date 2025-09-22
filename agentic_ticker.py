#!/usr/bin/env python3
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
                def on_event(e):
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
                price_data, indicator_data, events, forecasts, report, company_info = [], [], [], [], [], None
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
                
                st.session_state.price_data = price_data
                st.session_state.indicator_data = indicator_data
                st.session_state.events = events
                st.session_state.forecasts = forecasts
                st.session_state.report = report
                st.session_state.company_info = company_info
            except Exception as e:
                status.update(label="Agent loop failed", state="error")
                st.error(str(e))

    with col2:
        # Check if we have all the required data in session state
        has_results = all(key in st.session_state for key in ['price_data', 'indicator_data', 'events', 'forecasts', 'report'])
        
        if has_results and st.session_state.report:
            # Display unified header with company name and ticker
            company_info = st.session_state.get('company_info', {})
            company_name = company_info.get('company_name', ticker) if company_info else ticker
            st.header(f"📊 Results for {company_name} ({ticker})")
            st.markdown("---")  # Separator line
            
            st.subheader("📈 Price Chart")
            try:
                price_fig = create_price_chart(st.session_state.price_data, st.session_state.indicator_data)
                st.plotly_chart(price_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Chart unavailable: {e}")
            
            st.subheader("🔮 Price Forecasts")
            if 'forecasts' in st.session_state:
                try:
                    forecast_fig = create_forecast_chart(st.session_state.forecasts)
                    st.plotly_chart(forecast_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Forecast chart unavailable: {e}")
            
            if st.session_state.report and isinstance(st.session_state.report, dict) and 'content' in st.session_state.report:
                st.markdown(st.session_state.report["content"], unsafe_allow_html=True)
        else:
            st.info("Enter parameters and click Analyze to run the agent loop.")


if __name__ == "__main__":
    main()