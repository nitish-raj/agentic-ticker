#!/usr/bin/env python3
"""
Test script to exactly mimic the orchestrator flow and see what's happening
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ticker import Orchestrator

def test_flow_step_by_step():
    """Test the orchestrator flow step by step"""
    print("Creating orchestrator...")
    orch = Orchestrator()
    
    # Initialize transcript exactly like the orchestrator does
    goal = "Analyze AAPL by running five steps in order: load_prices(days=10), compute_indicators, detect_events(threshold=2.0), fetch_news(days=7), build_report."
    orch.transcript = [{"role": "user", "content": goal}]
    print(f"Initial transcript: {len(orch.transcript)} entries")
    
    # Step 1: Plan
    print("\nStep 1: Planning...")
    try:
        action = orch.planner.plan(orch.tools_spec(), goal, orch.transcript)
        print(f"Planner action: {action}")
    except Exception as e:
        print(f"Planner failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Execute load_prices
    if action.call:
        call = action.call
        name = call.get("name")
        args = call.get("args", {})
        print(f"\nStep 2: Executing {name} with args {args}")
        if name in orch.tools:
            res = orch.tools[name](**args) if args else orch.tools[name]()
            print(f"Result: {len(res) if isinstance(res, list) else type(res).__name__} items")
            
            # Apply our summarization
            transcript_res = res
            if isinstance(res, list) and len(res) > 5:
                transcript_res = {
                    "summary": f"List with {len(res)} items",
                    "item_type": type(res[0]).__name__ if res else "unknown",
                    "sample_keys": list(res[0].keys()) if res and isinstance(res[0], dict) else None
                }
            elif isinstance(res, dict) and len(res) > 5:
                transcript_res = {
                    "summary": f"Dict with {len(res)} keys: {list(res.keys())[:5]}...",
                    "size": len(res)
                }
            
            # Add to transcript like the orchestrator does
            orch.transcript.append({"role": "assistant", "content": {"call": {"name": name, "args": args}}})
            orch.transcript.append({"role": "tool", "name": name, "content": transcript_res})
            print(f"Transcript now has {len(orch.transcript)} entries")
            print(f"Last entry: {orch.transcript[-1]}")
        else:
            print(f"Unknown tool: {name}")
            return False
    
    # Step 3: Plan again
    print("\nStep 3: Planning again...")
    try:
        action2 = orch.planner.plan(orch.tools_spec(), goal, orch.transcript)
        print(f"Planner action: {action2}")
    except Exception as e:
        print(f"Planner failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_flow_step_by_step()
    sys.exit(0 if success else 1)