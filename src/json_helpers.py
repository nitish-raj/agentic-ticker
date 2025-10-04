import json
import re


def _json_safe(obj):
    import pandas as _pd
    import numpy as _np
    from datetime import datetime as _dt

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (_dt, _pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    return str(obj)


def _dumps(obj) -> str:
    return json.dumps(_json_safe(obj), ensure_ascii=False)


def _format_json_for_display(obj) -> str:
    """
    Format JSON data for readable display in the UI.
    Returns a formatted string with proper indentation and structure.
    Truncates large arrays to keep display compact.
    """
    if not obj or isinstance(obj, (str, int, float, bool)):
        return str(obj)

    try:
        # Create a truncated version for display
        truncated_obj = _truncate_large_data(obj)
        formatted = json.dumps(_json_safe(truncated_obj), indent=2, ensure_ascii=False)
        return formatted
    except Exception:
        return str(obj)


def _truncate_large_data(obj, max_array_items=3, max_string_length=50):
    """
    Truncate large data structures to keep display compact.
    """
    if isinstance(obj, dict):
        truncated = {}
        for key, value in obj.items():
            truncated[key] = _truncate_large_data(
                value, max_array_items, max_string_length
            )
        return truncated
    elif isinstance(obj, list):
        if len(obj) <= max_array_items:
            return [
                _truncate_large_data(item, max_array_items, max_string_length)
                for item in obj
            ]
        else:
            # Show first few items and indicate truncation
            truncated = [
                _truncate_large_data(item, max_array_items, max_string_length)
                for item in obj[:max_array_items]
            ]
            truncated.append(f"... {len(obj) - max_array_items} more items ...")
            return truncated
    elif isinstance(obj, str):
        if len(obj) > max_string_length:
            return obj[:max_string_length] + "..."
        return obj
    else:
        return obj


def _extract_json_text(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        end = s.rfind("```")
        if end != -1:
            first_nl = s.find("\n")
            if first_nl != -1:
                s = s[first_nl + 1 : end]
    s = s.strip()
    if s.lower().startswith("json"):
        s = s[4:].strip()
    return s


def _clean_trailing_commas(s: str) -> str:
    pattern = re.compile(r",\s*([}\]])")
    return pattern.sub(lambda m: m.group(1), s)


def _parse_json_strictish(text: str) -> dict:
    raw = _extract_json_text(text)
    left, right = raw.find("{"), raw.rfind("}")
    candidate = raw[left : right + 1] if (left >= 0 and right > left) else raw
    candidate = _clean_trailing_commas(candidate)
    return json.loads(candidate)
