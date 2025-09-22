from typing import List, Dict, Any, Optional
from .planner import GeminiPlanner
from .services import (
    validate_ticker,
    validate_ticker_gemini_only,
    validate_ticker_with_web_search,
    get_company_info,
    load_prices,
    compute_indicators,
    detect_events,
    forecast_prices,
    build_report
)
from .json_helpers import _format_json_for_display


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
                    on_event({"type": "call", "step": 6, "name": "web_search_ticker", "args": {"input_text": ticker_input}})
                
                try:
                    validated_ticker = validate_ticker_with_web_search(ticker_input)
                    
                    if validated_ticker:
                        steps.append({"type": "result", "name": "web_search_ticker", "result": validated_ticker})
                        if on_event:
                            on_event({"type": "result", "step": 6, "name": "web_search_ticker", "result": validated_ticker})
                            if validated_ticker.upper() != ticker_input.upper():
                                on_event({"type": "info", "message": f"Web search found ticker: {validated_ticker}"})
                    else:
                        error_msg = f"No valid ticker found for '{ticker_input}' after web search. Please check the company name or ticker symbol."
                        if on_event:
                            on_event({"type": "error", "step": 6, "name": "web_search_ticker", "error": error_msg})
                        raise ValueError(error_msg)
                except Exception as e:
                    if on_event:
                        on_event({"type": "error", "step": 6, "name": "web_search_ticker", "error": str(e)})
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
        
        # Step 1: Get company info
        if on_event:
            on_event({"type": "call", "step": 6, "name": "get_company_info", "args": {"ticker": validated_ticker}})
        
        try:
            company_info = get_company_info(validated_ticker)
            steps.append({"type": "result", "name": "get_company_info", "result": company_info})
            if on_event:
                on_event({"type": "result", "step": 6, "name": "get_company_info", "result": company_info})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 6, "name": "get_company_info", "error": str(e)})
            # Continue even if company info fails - not critical
            company_info = {"ticker": validated_ticker, "company_name": validated_ticker, "short_name": validated_ticker}
        
        # Step 2: Load prices
        if on_event:
            on_event({"type": "call", "step": 6, "name": "load_prices", "args": {"ticker": validated_ticker, "days": days}})
        
        try:
            price_data = load_prices(validated_ticker, days)
            if not price_data:
                error_msg = f"No price data found for ticker '{validated_ticker}'. The ticker may be delisted or invalid."
                if on_event:
                    on_event({"type": "error", "step": 6, "name": "load_prices", "error": error_msg})
                raise ValueError(error_msg)
            
            steps.append({"type": "result", "name": "load_prices", "result": price_data})
            if on_event:
                on_event({"type": "result", "step": 6, "name": "load_prices", "result": price_data})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 6, "name": "load_prices", "error": str(e)})
            raise
        
        # Step 3: Compute indicators
        if on_event:
            on_event({"type": "call", "step": 6, "name": "compute_indicators", "args": {"price_data": price_data}})
        
        try:
            indicator_data = compute_indicators(price_data)
            steps.append({"type": "result", "name": "compute_indicators", "result": indicator_data})
            if on_event:
                on_event({"type": "result", "step": 6, "name": "compute_indicators", "result": indicator_data})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 6, "name": "compute_indicators", "error": str(e)})
            raise
        
        # Step 3: Detect events
        if on_event:
            on_event({"type": "call", "step": 6, "name": "detect_events", "args": {"indicator_data": indicator_data, "threshold": threshold}})
        
        try:
            events = detect_events(indicator_data, threshold)
            steps.append({"type": "result", "name": "detect_events", "result": events})
            if on_event:
                on_event({"type": "result", "step": 6, "name": "detect_events", "result": events})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 6, "name": "detect_events", "error": str(e)})
            raise
        
        # Step 4: Forecast prices
        if on_event:
            on_event({"type": "call", "step": 6, "name": "forecast_prices", "args": {"indicator_data": indicator_data, "days": forecast_days}})
        
        try:
            forecasts = forecast_prices(indicator_data, forecast_days)
            steps.append({"type": "result", "name": "forecast_prices", "result": forecasts})
            if on_event:
                on_event({"type": "result", "step": 6, "name": "forecast_prices", "result": forecasts})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 6, "name": "forecast_prices", "error": str(e)})
            raise
        
        # Step 5: Build report
        if on_event:
            on_event({"type": "call", "step": 6, "name": "build_report", "args": {"ticker": validated_ticker, "events": events, "forecasts": forecasts, "company_info": company_info}})
        
        try:
            report = build_report(validated_ticker, events, forecasts, company_info)
            steps.append({"type": "result", "name": "build_report", "result": report})
            if on_event:
                on_event({"type": "result", "step": 6, "name": "build_report", "result": report})
                on_event({"type": "final", "data": report})
        except Exception as e:
            if on_event:
                on_event({"type": "error", "step": 6, "name": "build_report", "error": str(e)})
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


