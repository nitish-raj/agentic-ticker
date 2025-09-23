import os
import json
import requests
from typing import List, Dict, Any
from .json_helpers import _dumps, _parse_json_strictish
from .data_models import PlannerJSON, FunctionCall


class GeminiPlanner:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required")

    def _sanitize_error_message(self, error_msg: str) -> str:
        """Sanitize error messages to remove sensitive information like API keys"""
        import re
        # Pattern to match API keys in URLs
        api_key_pattern = r'key=[^&\s]+'
        sanitized = re.sub(api_key_pattern, 'key=[REDACTED]', error_msg)
        return sanitized

    def plan(self, tools_spec: List[Dict[str, Any]], goal: str, transcript: List[Dict[str, Any]], days: int = 30, threshold: float = 2.0, forecast_days: int = 7) -> PlannerJSON:
        system = (
            "You are an intelligent stock analysis orchestrator. Your job is to analyze the available functions, "
            "understand the current state from the transcript, and decide the next function to call to complete the stock analysis. "
            "\n\n"
            "USER PARAMETERS:\n"
            f"- days: {days} (analysis period for historical price data)\n"
            f"- threshold: {threshold} (threshold for detecting significant events)\n"
            f"- forecast_days: {forecast_days} (number of days to forecast)\n"
            "\n"
            "ANALYSIS SEQUENCE LOGIC:\n"
            "Follow this exact sequence in order. Do NOT skip any steps:\n"
            "1. validate_ticker_gemini_only: Use input_text=goal (the user's ticker_input)\n"
            "2. If step 1 fails, use validate_ticker_with_web_search: Use input_text=goal (the user's ticker_input)\n"
            "3. get_company_info: Use ticker=validated_ticker (from step 1 or 2)\n"
            "4. load_prices: Use ticker=validated_ticker and days={days}\n"
            "5. compute_indicators: Use the price data from step 4\n"
            "6. detect_events: Use indicator_data=compute_indicators (from step 5) and threshold={threshold}\n"
            "7. forecast_prices: Use indicator_data=compute_indicators (from step 5) and days={forecast_days}\n"
            "8. build_report: Use ticker=validated_ticker, events=detect_events (from step 6), forecasts=forecast_prices (from step 7), and company_info=get_company_info (from step 3)\n"
            "\n"
            "CRITICAL: You MUST execute ALL steps in order. Do NOT jump ahead or skip steps.\n"
            "\n"
            "CONTEXT AWARENESS:\n"
            "- Check the transcript to see what data is already available\n"
            "- Use results from previous steps as arguments for subsequent functions\n"
            "- The ticker_input contains the user's input (company name, ticker, or description)\n"
            "- NEVER skip steps unless you see EXPLICIT results in the transcript\n"
            "- If you don't see a function's result in the transcript, you MUST call that function\n"
            "- Example: If you don't see 'detect_events' result, you MUST call detect_events\n"
            "- IMPORTANT: Before calling build_report, ensure you have: validated_ticker, detect_events, forecast_prices, and get_company_info results\n"
            "- build_report is the FINAL step - call it only when all other data is available\n"
            "\n"
            "ARGUMENT HANDLING:\n"
            "- Extract function arguments from docstrings and previous results\n"
            "- Use exact parameter names as specified in function signatures\n"
            "- Pass validated ticker from step 1 to subsequent functions\n"
            "- For validate_ticker_gemini_only, use input_text=goal (the user's ticker_input)\n"
            "- For validate_ticker_with_web_search, use input_text=goal (the user's ticker_input)\n"
            f"- For load_prices, use ticker=validated_ticker and days={days}\n"
            f"- For detect_events, use indicator_data=context_key and threshold={threshold}\n"
            f"- For forecast_prices, use indicator_data=context_key and days={forecast_days}\n"
            "- For build_report, use ticker=validated_ticker, events=detect_events, forecasts=forecast_prices, and company_info=get_company_info\n"
            "- IMPORTANT: For large data structures like indicator_data, events, forecasts, use the context key name as argument value\n"
            "- Example: if context has 'compute_indicators' result, use 'compute_indicators' as indicator_data argument\n"
            "\n"
            "OUTPUT FORMAT:\n"
            "Only output a single JSON object with either {\"call\":{name,args}} or {\"final\":\"message string\"}.\n"
            "The final field must be a string, not a dict. Use exact argument names from the functions' docstrings."
        )
        payload_text = _dumps({"tools": tools_spec, "ticker_input": goal, "transcript": transcript})
        url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"
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


