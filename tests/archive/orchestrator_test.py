#!/usr/bin/env python3
"""
Test script to mimic exactly what the orchestrator does
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ticker import load_prices, compute_indicators

def test_orchestrator_flow():
    """Test the exact flow that the orchestrator does"""
    print("Step 1: load_prices")
    try:
        price_data = load_prices("AAPL", days=30)
        print(f"load_prices returned {len(price_data)} items")
    except Exception as e:
        print(f"load_prices failed: {e}")
        return False
    
    print("\nStep 2: compute_indicators")
    try:
        # This is exactly what the orchestrator does
        res = compute_indicators(price_data)
        print(f"compute_indicators returned {len(res)} items")
    except Exception as e:
        print(f"compute_indicators failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_orchestrator_flow()
    sys.exit(0 if success else 1)