import os
import json
import requests
from typing import List, Dict, Any
try:
    from json_helpers import _dumps, _parse_json_strictish
    from data_models import PlannerJSON, FunctionCall
except ImportError:
    from .json_helpers import _dumps, _parse_json_strictish
    from .data_models import PlannerJSON, FunctionCall

# Import configuration system
try:
    from .config import get_config
except ImportError:
    # Fallback for when running as script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import get_config

# Import decorators
try:
    from decorators import handle_errors, log_execution, time_execution, validate_inputs, retry_on_failure
except ImportError:
    # Fallback for development environment
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.decorators import handle_errors, log_execution, time_execution, validate_inputs, retry_on_failure


class GeminiPlanner:
    def __init__(self):
        try:
            config = get_config()
            gemini_config = config.gemini
            self.api_key = gemini_config.api_key
            self.model = gemini_config.model
            self.api_base = gemini_config.api_base
        except Exception:
            # Fallback if config not available
            self.api_key = ""
            self.model = "gemini-2.5-flash-lite"
            self.api_base = "https://generativelanguage.googleapis.com/v1beta"
        
        # Allow test environment with mock key
        if not self.api_key and not os.environ.get('PYTEST_CURRENT_TEST'):
            raise RuntimeError("GEMINI_API_KEY is required in config.yaml")
        elif not self.api_key and os.environ.get('PYTEST_CURRENT_TEST'):
            # Use mock key for testing
            self.api_key = "test_mock_key_for_pytest_only"

    def _sanitize_error_message(self, error_msg: str) -> str:
        """Sanitize error messages to remove sensitive information like API keys"""
        try:
            from sanitization import sanitize_error_message
        except ImportError:
            from .sanitization import sanitize_error_message
        return sanitize_error_message(error_msg)

    def plan(self, tools_spec: List[Dict[str, Any]], goal: str, transcript: List[Dict[str, Any]], days: int = 30, threshold: float = 2.0, forecast_days: int = 7, asset_type: str = "ambiguous") -> PlannerJSON:
        system = (
            "You are an intelligent financial asset analysis orchestrator. Your job is to analyze the available functions, "
            "understand the current state from the transcript, classify the asset type, and decide the next function to call to complete the analysis. "
            "\n\n"
            "USER PARAMETERS:\n"
            f"- days: {days} (analysis period for historical price data)\n"
            f"- threshold: {threshold} (threshold for detecting significant events)\n"
            f"- forecast_days: {forecast_days} (number of days to forecast)\n"
            f"- asset_type: {asset_type} (asset type: stock, crypto, or ambiguous)\n"
            "\n"
            "ASSET CLASSIFICATION (FIRST STEP):\n"
            "Before any analysis, you MUST classify the user's input as one of these asset types:\n"
            "- 'stock': Traditional stock/company (e.g., 'AAPL', 'Apple', 'Microsoft', 'GOOGL')\n"
            "- 'crypto': Cryptocurrency (e.g., 'BTC', 'Bitcoin', 'ETH', 'Ethereum', 'DOGE', 'Dogecoin')\n"
            "- 'ambiguous': Cannot determine from input (e.g., 'TSLA' could be Tesla stock or a crypto token)\n"
            "\n"
            "THINKING PROCESS (REQUIRED):\n"
            "Before calling any function, you MUST explain your reasoning in this format:\n"
            "THINKING: I need to analyze the user's input '{goal}' to determine the asset type. "
            "Based on the input, I classify this as [stock/crypto/ambiguous] because [brief reasoning]. "
            "I will now proceed with the appropriate analysis sequence.\n"
            "\n"
            "ANALYSIS SEQUENCE LOGIC:\n"
            "Based on asset classification, follow the appropriate sequence:\n"
            "\n"
            "FOR STOCK ASSETS:\n"
            "1. validate_ticker: Use input_text=goal (the user's ticker_input)\n"
            "3. get_company_info: Use ticker=validated_ticker (from step 1 or 2)\n"
            "4. load_prices: Use ticker=validated_ticker and days={days}\n"
            "5. compute_indicators: Use the price data from step 4\n"
            "6. detect_events: Use indicator_data=compute_indicators (from step 5) and threshold={threshold}\n"
            "7. forecast_prices: Use indicator_data=compute_indicators (from step 5) and days={forecast_days}\n"
            "8. build_report: Use ticker=validated_ticker, events=detect_events (from step 6), forecasts=forecast_prices (from step 7), company_info=get_company_info (from step 3), price_data=load_prices (from step 4), and indicator_data=compute_indicators (from step 5)\n"
            "\n"
            "FOR CRYPTO ASSETS:\n"
            "1. validate_ticker: Use input_text=goal (the user's crypto_input)\n"
            "2. get_crypto_info: Use ticker=validated_ticker (from step 1) and original_input=goal (the original user input)\n"
            "3. load_prices: Use ticker=validated_ticker and days={days}\n"
            "4. compute_indicators: Use the crypto price data from step 3\n"
            "5. detect_events: Use indicator_data=compute_indicators (from step 4) and threshold={threshold}\n"
            "6. forecast_prices: Use indicator_data=compute_indicators (from step 4) and days={forecast_days}\n"
            "7. build_report: Use ticker=validated_ticker, events=detect_events (from step 5), forecasts=forecast_prices (from step 6), crypto_info=get_crypto_info (from step 2), price_data=load_prices (from step 3), and indicator_data=compute_indicators (from step 4)\n"
            "\n"
            "FOR AMBIGUOUS ASSETS:\n"
            "1. ddgs_search: Use query=goal (the user's input) to find information about the company/asset\n"
            "2. validate_ticker: Use input_text=goal (the user's input) - this will now use web search results to find the correct ticker\n"
            "3. If validation returns a stock ticker (e.g., 'AAPL'), proceed with STOCK sequence from step 2\n"
            "4. If validation returns a crypto ticker (e.g., 'BTC-USD'), proceed with CRYPTO sequence from step 2\n"
            "5. If validation failed, return final error message\n"
            "\n"
            "CRITICAL: You MUST execute ALL steps in order for your chosen sequence. Do NOT jump ahead or skip steps.\n"
            "\n"
            "CONTEXT AWARENESS:\n"
            "- Check the transcript to see what data is already available\n"
            "- Use results from previous steps as arguments for subsequent functions\n"
            "- The ticker_input contains the user's input (company name, ticker, crypto symbol, or description)\n"
            "- NEVER skip steps unless you see EXPLICIT results in the transcript\n"
            "- If you don't see a function's result in the transcript, you MUST call that function\n"
            "- Example: If you don't see 'detect_events' result, you MUST call detect_events\n"
            "- IMPORTANT: Before calling build_report or build_crypto_report, ensure you have: validated_ticker/crypto_id, detect_events, forecast_prices, and company_info/crypto_info results\n"
            "- build_report/build_crypto_report is the FINAL step - call it only when all other data is available\n"
            "- CRITICAL: Always check that required data exists in context before using it as arguments\n"
            "- If context data is missing, call the appropriate function to generate it first\n"
            "\n"
            "ARGUMENT HANDLING:\n"
            "- Extract function arguments from docstrings and previous results\n"
            "- Use exact parameter names as specified in function signatures\n"
            "- Pass validated ticker from validation steps to subsequent functions\n"
            "- For validation functions, use input_text=goal (the user's input)\n"
            f"- For load_prices, use ticker=validated_ticker and days={days}\n"
            f"- For detect_events, use indicator_data=context_key and threshold={threshold}\n"
            f"- For forecast_prices, use indicator_data=context_key and days={forecast_days}\n"
            "- For build_report, use ticker=validated_ticker, events=detect_events, forecasts=forecast_prices, company_info=get_company_info, crypto_info=get_crypto_info (if available), price_data=load_prices, and indicator_data=compute_indicators\n"
            "- For get_crypto_info, use ticker=validated_ticker and original_input=goal (the original user input)\n"
            "- IMPORTANT: For large data structures like indicator_data, events, forecasts, use the context key name as argument value\n"
            "- Example: if context has 'compute_indicators' result, use 'compute_indicators' as indicator_data argument\n"
            "- CRITICAL: NEVER pass empty strings ('') as arguments. If you don't have a value, either:\n"
            "  1. Use a context reference (e.g., 'validated_ticker', 'detect_events')\n"
            "  2. Use the function's default value if available\n"
            "  3. Omit the parameter entirely if it has a default\n"
            "  4. For required parameters without defaults, you MUST have a valid value from context\n"
            "\n"
            "OUTPUT FORMAT:\n"
            "Only output a single JSON object with either {\"call\":{name,args}} or {\"final\":\"message string\"}.\n"
            "The final field must be a string, not a dict. Use exact argument names from the functions' docstrings."
        )
        payload_text = _dumps({"tools": tools_spec, "ticker_input": goal, "transcript": transcript})
        url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"
        # Sanitize URL for any potential logging/debug output
        try:
            from sanitization import sanitize_url
        except ImportError:
            from .sanitization import sanitize_url
        sanitized_url = sanitize_url(url)
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": payload_text}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }
        try:
            r = requests.post(url, json=body, timeout=120)
            r.raise_for_status()
        except requests.exceptions.Timeout as e:
            sanitized_error = self._sanitize_error_message(str(e))
            raise RuntimeError(f"API request timed out after 120 seconds: {sanitized_error}") from e
        except requests.exceptions.RequestException as e:
            sanitized_error = self._sanitize_error_message(str(e))
            raise RuntimeError(f"API request failed: {sanitized_error}") from e
        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Invalid Gemini response: {data}") from e
        try:
            obj = _parse_json_strictish(text)
            # Convert dict call to FunctionCall if needed
            if isinstance(obj.get('call'), dict):
                obj['call'] = FunctionCall(**obj['call'])
            # Convert dict final to string if needed
            if isinstance(obj.get('final'), dict):
                obj['final'] = str(obj['final'])
        except Exception as ex:
            repair_body = {
                "system_instruction": {"parts": [{"text": system + " Return ONLY strict JSON with double quotes, no comments, no trailing commas."}]},
                "contents": [{"role": "user", "parts": [{"text": payload_text}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
            }
            try:
                rr = requests.post(url, json=repair_body, timeout=120)
                rr.raise_for_status()
            except requests.exceptions.Timeout as e:
                sanitized_error = self._sanitize_error_message(str(e))
                raise RuntimeError(f"API request timed out after 120 seconds (repair attempt): {sanitized_error}") from e
            except requests.exceptions.RequestException as e:
                sanitized_error = self._sanitize_error_message(str(e))
                raise RuntimeError(f"API request failed (repair attempt): {sanitized_error}") from e
            d2 = rr.json()
            try:
                text2 = d2["candidates"][0]["content"]["parts"][0]["text"]
                obj = _parse_json_strictish(text2)
            except Exception:
                snippet = (text or "")[:300]
                raise RuntimeError(f"Planner JSON parse failed. First attempt snippet: {snippet}") from ex
        return PlannerJSON(**obj)


