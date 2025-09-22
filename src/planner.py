import os
import json
import requests
from typing import List, Dict, Any
from .json_helpers import _dumps, _parse_json_strictish
from .data_models import PlannerJSON


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


