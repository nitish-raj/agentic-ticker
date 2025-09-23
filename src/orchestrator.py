from typing import List, Dict, Any, Optional
from .planner import GeminiPlanner
from .services import (
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

try:
    import streamlit as st
except ImportError:
    st = None


class Orchestrator:
    def __init__(self):
        self.planner = GeminiPlanner()
        self.tools = {
            "validate_ticker_gemini_only": validate_ticker_gemini_only,
            "validate_ticker_with_web_search": validate_ticker_with_web_search,
            "get_company_info": get_company_info,
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

    def _sanitize_error_message(self, error_msg: str) -> str:
        """Sanitize error messages to remove sensitive information like API keys"""
        # Remove API keys from error messages
        import re
        # Pattern to match API keys in URLs
        api_key_pattern = r'key=[^&\s]+'
        sanitized = re.sub(api_key_pattern, 'key=[REDACTED]', error_msg)
        return sanitized

    def run(self, ticker_input: str, days: int, threshold: float, forecast_days: int, on_event: Optional[callable] = None) -> List[Dict[str, Any]]:
        # Gemini-orchestrated execution
        steps: List[Dict[str, Any]] = []
        context: Dict[str, Any] = {
            "ticker_input": ticker_input,
            "days": days,
            "threshold": threshold,
            "forecast_days": forecast_days
        }
        
        if on_event:
            on_event({"type": "info", "message": f"Starting Gemini-orchestrated analysis for '{ticker_input}'..."})
        
        step_count = 0
        max_steps = 10  # Prevent infinite loops
        
        while step_count < max_steps:
            step_count += 1
            
            # Get tools specification and transcript for planning
            tools_spec = self.tools_spec()
            transcript = steps.copy()
            
            # Add context summary to transcript for Gemini to understand available data
            # Don't pass large data structures directly, just reference them by key
            for key, value in context.items():
                if value and key not in ["ticker_input"]:  # Don't include original input in transcript
                    if isinstance(value, list) and len(value) > 0:
                        transcript.append({"type": "context", "key": key, "value": f"{type(value).__name__} with {len(value)} items"})
                    elif isinstance(value, dict):
                        transcript.append({"type": "context", "key": key, "value": f"dict with keys: {list(value.keys())[:3]}"})
                    else:
                        transcript.append({"type": "context", "key": key, "value": str(value)[:50]})
            
            if on_event:
                on_event({"type": "planning", "step": step_count})
            
            try:
                # Let Gemini decide the next action - pass all UI parameters
                plan = self.planner.plan(tools_spec, ticker_input, transcript, days, threshold, forecast_days)
                
                if plan.final:
                    if on_event:
                        on_event({"type": "final", "data": plan.final})
                    break
                
                if plan.call and plan.call.name:
                    func_name = plan.call.name
                    func_args = plan.call.args or {}
                    
                    if on_event:
                        on_event({"type": "call", "step": step_count, "name": func_name, "args": func_args})
                    
                    if func_name in self.tools:
                        try:
                            # Process arguments - replace context references with actual data
                            processed_args = {}
                            for arg_name, arg_value in func_args.items():
                                if isinstance(arg_value, str) and arg_value in context:
                                    # If argument is a context key, use the actual data
                                    processed_args[arg_name] = context[arg_value]
                                elif isinstance(arg_value, str):
                                    # Try to convert string numbers to proper types
                                    try:
                                        if '.' in arg_value:
                                            processed_args[arg_name] = float(arg_value)
                                        else:
                                            processed_args[arg_name] = int(arg_value)
                                    except ValueError:
                                        processed_args[arg_name] = arg_value
                                else:
                                    processed_args[arg_name] = arg_value
                            
                            # Execute the function with processed arguments
                            result = self.tools[func_name](**processed_args)
                            
                            # Store result in context for future steps
                            context[func_name] = result
                            
                            steps.append({"type": "result", "name": func_name, "result": result})
                            if on_event:
                                on_event({"type": "result", "step": step_count, "name": func_name, "result": result})
                            
                            # Special handling for ticker validation
                            if func_name == "validate_ticker_gemini_only" and result:
                                context["validated_ticker"] = result
                                if result.upper() != ticker_input.upper() and on_event:
                                    on_event({"type": "info", "message": f"Converted '{ticker_input}' to ticker symbol: {result}"})
                            elif func_name == "validate_ticker_with_web_search" and result:
                                context["validated_ticker"] = result
                                if on_event:
                                    on_event({"type": "info", "message": f"Web search found ticker: {result}"})
                            
                        except Exception as e:
                            error_msg = f"Error executing {func_name}: {str(e)}"
                            if on_event:
                                on_event({"type": "error", "step": step_count, "name": func_name, "error": error_msg})
                            raise
                    else:
                        error_msg = f"Unknown function: {func_name}"
                        if on_event:
                            on_event({"type": "error", "step": step_count, "name": func_name, "error": error_msg})
                        raise ValueError(error_msg)
                        
            except Exception as e:
                error_msg = self._sanitize_error_message(str(e))
                if on_event:
                    on_event({"type": "error", "step": step_count, "name": "planning", "error": error_msg})
                raise
        
        # Store results in session state for UI
        if st is not None:
            try:
                if context.get("load_prices"):
                    st.session_state.price_data = context["load_prices"]
                if context.get("compute_indicators"):
                    st.session_state.indicator_data = context["compute_indicators"]
                if context.get("detect_events"):
                    st.session_state.events = context["detect_events"]
                if context.get("forecast_prices"):
                    st.session_state.forecasts = context["forecast_prices"]
                if context.get("build_report"):
                    st.session_state.report = context["build_report"]
                if context.get("get_company_info"):
                    st.session_state.company_info = context["get_company_info"]
            except:
                pass  # Session state not available
        
        return steps


