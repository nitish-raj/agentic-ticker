#!/usr/bin/env python3
"""
Debug the orchestrator to understand the data reference issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ticker import Orchestrator

def debug_orchestrator():
    """Debug the orchestrator step by step"""
    print("Creating orchestrator...")
    orch = Orchestrator()
    
    # Test with a simple goal
    goal = "Analyze AAPL by running five steps in order: load_prices(days=10), compute_indicators, detect_events(threshold=2.0), fetch_news(days=7), build_report."
    
    print("Starting orchestrator...")
    try:
        def on_event(e):
            print(f"Event: {e['type']}")
            if e['type'] == 'call':
                print(f"  Calling {e['name']} with args {e['args']}")
                # Print the actual args that will be passed to the function
                fixed_args = {}
                for arg_name, arg_value in e['args'].items():
                    if isinstance(arg_value, str) and arg_value.startswith("data_ref:"):
                        data_id = arg_value.split(":", 1)[1]
                        if data_id in orch.data_cache:
                            fixed_args[arg_name] = orch.data_cache[data_id]
                            print(f"    Resolving {arg_name}: {arg_value} -> {type(orch.data_cache[data_id])} with {len(orch.data_cache[data_id]) if hasattr(orch.data_cache[data_id], '__len__') else 'N/A'} items")
                        else:
                            fixed_args[arg_name] = arg_value
                            print(f"    Could not resolve {arg_name}: {arg_value}")
                    else:
                        fixed_args[arg_name] = arg_value
                print(f"  Fixed args: {fixed_args}")
            elif e['type'] == 'result':
                result = e['result']
                if isinstance(result, list):
                    print(f"  Result: List with {len(result)} items")
                elif isinstance(result, dict):
                    print(f"  Result: Dict with keys {list(result.keys())}")
                else:
                    print(f"  Result: {type(result).__name__}")
                
                # Check what's being stored in transcript
                # This is a bit tricky to access, but let's see the last few transcript entries
                if hasattr(orch, 'data_cache'):
                    print(f"  Data cache size: {len(orch.data_cache)}")
                    for key, value in list(orch.data_cache.items())[-2:]:
                        print(f"    {key}: {type(value)} with {len(value) if hasattr(value, '__len__') else 'N/A'} items")
        
        steps = orch.run(goal, max_steps=10, on_event=on_event)
        print("Orchestrator completed successfully!")
        return True
    except Exception as e:
        print(f"Orchestrator failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_orchestrator()
    sys.exit(0 if success else 1)