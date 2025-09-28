from typing import List, Dict, Any, Optional, Callable
import re
import logging
from functools import wraps
from datetime import datetime

from .planner import GeminiPlanner
from .services import (
    validate_ticker,
    get_company_info,
    get_crypto_info,
    load_prices,
    load_crypto_prices,
    compute_indicators,
    detect_events,
    forecast_prices,
    build_report,
    ddgs_search
)
from .json_helpers import _json_safe, _format_json_for_display


# Handle optional streamlit import with proper typing
try:
    import streamlit as st
    st_available = True
except ImportError:
    st = None  # type: ignore
    st_available = False

# Configure logging
logger = logging.getLogger(__name__)

# Utility functions for error handling and sanitization
def sanitize_error_message(error_msg: str) -> str:
    """Sanitize error messages to remove sensitive information like API keys"""
    # Remove API keys from error messages
    api_key_pattern = r'key=[^&\s]+'
    sanitized = re.sub(api_key_pattern, 'key=[REDACTED]', error_msg)
    return sanitized

# Decorators for cross-cutting concerns
def handle_errors(func):
    """Decorator for consistent error handling across functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

def log_execution(func):
    """Decorator for logging function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed {func.__name__} in {duration:.2f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed {func.__name__} after {duration:.2f}s: {e}")
            raise
    return wrapper

# Utility functions for argument processing
def process_function_arguments(func_args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Process and validate function arguments, replacing context references"""
    processed_args = {}
    for arg_name, arg_value in func_args.items():
        if isinstance(arg_value, str) and arg_value in context:
            # If argument is a context key, use the actual data
            processed_args[arg_name] = context[arg_value]
        elif isinstance(arg_value, str):
            # Handle empty strings - don't skip them, let the function validation handle it
            # This ensures required parameters are passed even if empty, so validation can provide proper error messages
            if arg_value.strip() == '':
                processed_args[arg_name] = arg_value  # Pass empty string as-is
            else:
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
    return processed_args

# Utility functions for session state management
def update_session_state(context: Dict[str, Any]) -> None:
    """Update Streamlit session state with context data"""
    if not st_available or st is None:
        return
    
    try:
        # Handle both stock and crypto price data
        if context.get("load_prices"):
            st.session_state.price_data = context["load_prices"]
        elif context.get("load_crypto_prices"):
            st.session_state.price_data = context["load_crypto_prices"]
            
        if context.get("compute_indicators"):
            st.session_state.indicator_data = context["compute_indicators"]
        if context.get("detect_events"):
            st.session_state.events = context["detect_events"]
        if context.get("forecast_prices"):
            st.session_state.forecasts = context["forecast_prices"]
        if context.get("build_report"):
            st.session_state.report = context["build_report"]
            # Mark analysis as completed when report is built
            st.session_state.analysis_completed = True
            
        # Handle both company and crypto info
        if context.get("get_company_info"):
            st.session_state.company_info = context["get_company_info"]
        elif context.get("get_crypto_info"):
            st.session_state.crypto_info = context["get_crypto_info"]
            
    except Exception as e:
        logger.warning(f"Failed to update session state: {e}")

# Utility functions for function result messaging
def get_function_success_message(func_name: str, result: Any) -> str:
    """Generate appropriate success message based on function name and result"""
    if func_name == "ddgs_search" and result:
        count = len(result) if isinstance(result, list) else 1
        return f"✅ Web search completed: Found {count} relevant results. This will help me better understand the asset."
    elif func_name == "get_company_info" and result:
        company_name = result.get('company_name', 'Unknown') if isinstance(result, dict) else 'Unknown'
        return f"✅ Company info retrieved: Successfully gathered details for {company_name}."
    elif func_name == "get_crypto_info" and result:
        crypto_name = result.get('name', 'Unknown') if isinstance(result, dict) else 'Unknown'
        return f"✅ Crypto info retrieved: Successfully gathered details for {crypto_name}."
    elif func_name == "load_prices" and result:
        days_count = len(result) if isinstance(result, list) else 0
        return f"✅ Price data loaded: Successfully retrieved {days_count} days of historical price data."
    elif func_name == "compute_indicators" and result:
        return f"✅ Technical indicators computed: RSI, MACD, and Bollinger Bands calculated successfully."
    elif func_name == "detect_events" and result:
        events_count = len(result) if isinstance(result, list) else 0
        return f"✅ Events detected: Found {events_count} significant price events based on the threshold."
    elif func_name == "forecast_prices" and result:
        forecast_count = len(result) if isinstance(result, list) else 0
        return f"✅ Forecasts generated: Created {forecast_count} days of price predictions using ML models."
    elif func_name == "build_report" and result:
        return f"✅ Report built: Comprehensive analysis report completed with all findings."
    return f"✅ {func_name} completed successfully."

# Utility functions for context management
def build_context_summary(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a summary of context data for transcript"""
    transcript = []
    for key, value in context.items():
        if value and key not in ["ticker_input"]:  # Don't include original input in transcript
            if isinstance(value, list) and len(value) > 0:
                transcript.append({"type": "context", "key": key, "value": f"{type(value).__name__} with {len(value)} items"})
            elif isinstance(value, dict):
                transcript.append({"type": "context", "key": key, "value": f"dict with keys: {list(value.keys())[:3]}"})
            else:
                transcript.append({"type": "context", "key": key, "value": str(value)[:50]})
    return transcript


class Orchestrator:
    def __init__(self):
        self.planner = GeminiPlanner()
        self.tools = {
            "validate_ticker": validate_ticker,
            "get_company_info": get_company_info,
            "get_crypto_info": get_crypto_info,
            "load_prices": load_prices,
            "load_crypto_prices": load_crypto_prices,
            "compute_indicators": compute_indicators,
            "detect_events": detect_events,
            "forecast_prices": forecast_prices,
            "build_report": build_report,
            "ddgs_search": ddgs_search,
        }

    @handle_errors
    @log_execution
    def tools_spec(self) -> List[Dict[str, Any]]:
        spec = []
        for name, fn in self.tools.items():
            spec.append({"name": name, "docstring": fn.__doc__ or "", "signature": str(fn)})
        return spec

    @handle_errors
    @log_execution
    def run(self, ticker_input: str, days: int, threshold: float, forecast_days: int, on_event: Optional[Callable] = None) -> List[Dict[str, Any]]:
        # Gemini-orchestrated execution
        steps: List[Dict[str, Any]] = []
        
        context: Dict[str, Any] = {
            "ticker_input": ticker_input,
            "days": days,
            "threshold": threshold,
            "forecast_days": forecast_days,
            "asset_type": "ambiguous"  # Let Gemini determine this
        }
        
        if on_event:
            on_event({"type": "info", "message": f"🤖 Starting agentic analysis for '{ticker_input}'..."})
            on_event({"type": "info", "message": f"📝 Initial analysis: User wants to analyze '{ticker_input}'. Let me understand what this asset is and gather comprehensive information."})
        
        step_count = 0
        max_steps = 10  # Prevent infinite loops
        
        while step_count < max_steps:
            step_count += 1
            
            # Get tools specification and transcript for planning
            tools_spec = self.tools_spec()
            transcript = steps.copy()
            
            # Add context summary to transcript for Gemini to understand available data
            transcript.extend(build_context_summary(context))
            
            if on_event:
                on_event({"type": "planning", "step": step_count, "message": f"🧠 Planning step {step_count}..."})
                # Add strategic thinking before each step
                if step_count == 1:
                    on_event({"type": "info", "message": f"🧠 Strategic thinking: I need to first classify what type of asset '{ticker_input}' is, then gather the appropriate data for analysis."})
            
            try:
                # Let Gemini decide the next action - pass all UI parameters including asset type
                plan = self.planner.plan(tools_spec, ticker_input, transcript, days, threshold, forecast_days, context["asset_type"])
                
                if plan.final:
                    if on_event:
                        on_event({"type": "final", "data": plan.final})
                    break
                
                if plan.call and plan.call.name:
                    func_name = plan.call.name
                    func_args = plan.call.args or {}
                    
                    if on_event:
                        on_event({"type": "call", "step": step_count, "name": func_name, "args": func_args, "message": f"🔄 Calling {func_name}..."})
                    
                    if func_name in self.tools:
                        try:
                            # Process arguments - replace context references with actual data
                            processed_args = process_function_arguments(func_args, context)
                            
                            
                            
                            # Execute the function with processed arguments
                            result = self.tools[func_name](**processed_args)
                            
                            # Store result in context for future steps
                            context[func_name] = result
                            
                            # Add post-function call thinking to show what was learned
                            if result and on_event:
                                success_message = get_function_success_message(func_name, result)
                                on_event({"type": "info", "message": success_message})
                            
                            steps.append({"type": "result", "name": func_name, "result": result})
                            if on_event:
                                on_event({"type": "result", "step": step_count, "name": func_name, "result": result, "message": f"✅ {func_name} completed successfully"})
                            
                            # Special handling for ticker validation
                            if func_name == "validate_ticker" and result:
                                context["validated_ticker"] = result
                                if isinstance(result, str) and result.upper() != ticker_input.upper() and on_event:
                                    on_event({"type": "info", "message": f"Converted '{ticker_input}' to ticker symbol: {result}"})
                            
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
                error_msg = sanitize_error_message(str(e))
                if on_event:
                    on_event({"type": "error", "step": step_count, "name": "planning", "error": error_msg})
                raise
        
        # Store results in session state for UI
        update_session_state(context)
        
        return steps


