#!/usr/bin/env python3
"""
Debug test to understand the data reference issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ticker import Orchestrator, detect_events

def test_detect_events_with_string():
    """Test detect_events with a string argument to reproduce the error"""
    print("Testing detect_events with string argument...")
    try:
        # This should reproduce the error
        result = detect_events("data_ref:data_1", 2.0)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_detect_events_with_empty_list():
    """Test detect_events with an empty list"""
    print("\nTesting detect_events with empty list...")
    try:
        result = detect_events([], 2.0)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detect_events_with_string()
    test_detect_events_with_empty_list()