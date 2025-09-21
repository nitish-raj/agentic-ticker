#!/usr/bin/env python3
"""
Test script to verify the JSON truncation fix in agentic_ticker.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ticker import Orchestrator

def test_large_data_handling():
    """Test that the orchestrator can handle large data without JSON truncation"""
    print("Testing large data handling...")
    
    # Create an orchestrator instance
    orch = Orchestrator()
    
    # Test with a goal that would generate large data
    goal = "Analyze AAPL by running five steps in order: load_prices(days=30), compute_indicators, detect_events(threshold=2.0), fetch_news(days=7), build_report."
    
    try:
        # Run the orchestrator with event callback to see progress
        def on_event(e):
            print(f"Event: {e['type']}")
            if e['type'] == 'result':
                result = e['result']
                if isinstance(result, list):
                    print(f"  Result: List with {len(result)} items")
                elif isinstance(result, dict):
                    print(f"  Result: Dict with keys {list(result.keys())}")
        
        steps = orch.run(goal, max_steps=10, on_event=on_event)
        print("Test completed successfully!")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_large_data_handling()
    sys.exit(0 if success else 1)