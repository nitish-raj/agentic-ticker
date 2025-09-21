#!/usr/bin/env python3
"""
Test script to verify that the orchestrator now accepts user inputs
and doesn't use hardcoded values.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ticker import Orchestrator

def test_orchestrator_with_inputs():
    """Test that orchestrator accepts user inputs correctly"""
    print("Testing orchestrator with user inputs...")
    
    # Test with different ticker and parameters
    orch = Orchestrator()
    
    # Test 1: Different ticker (MSFT instead of hardcoded AAPL)
    print("\n1. Testing with MSFT ticker...")
    try:
        steps = orch.run(ticker="MSFT", days=30, threshold=2.0, forecast_days=5)
        print("✓ Orchestrator accepted MSFT ticker")
        
        # Check if the steps contain the correct ticker
        for step in steps:
            if step['type'] == 'call' and step['name'] == 'load_prices':
                args = step.get('args', {})
                if args.get('ticker') == 'MSFT':
                    print("✓ Correct ticker passed to load_prices")
                else:
                    print(f"✗ Wrong ticker: {args.get('ticker')}")
                    return False
                    
            if step['type'] == 'call' and step['name'] == 'build_report':
                args = step.get('args', {})
                if args.get('ticker') == 'MSFT':
                    print("✓ Correct ticker passed to build_report")
                else:
                    print(f"✗ Wrong ticker in build_report: {args.get('ticker')}")
                    return False
                    
    except Exception as e:
        print(f"✗ Error with MSFT: {e}")
        return False
    
    # Test 2: Different parameters
    print("\n2. Testing with different parameters...")
    try:
        steps = orch.run(ticker="GOOGL", days=60, threshold=3.0, forecast_days=10)
        print("✓ Orchestrator accepted different parameters")
        
        # Check if the steps contain the correct parameters
        for step in steps:
            if step['type'] == 'call' and step['name'] == 'load_prices':
                args = step.get('args', {})
                if args.get('days') == 60:
                    print("✓ Correct days parameter passed")
                else:
                    print(f"✗ Wrong days: {args.get('days')}")
                    return False
                    
            if step['type'] == 'call' and step['name'] == 'detect_events':
                args = step.get('args', {})
                if args.get('threshold') == 3.0:
                    print("✓ Correct threshold parameter passed")
                else:
                    print(f"✗ Wrong threshold: {args.get('threshold')}")
                    return False
                    
            if step['type'] == 'call' and step['name'] == 'forecast_prices':
                args = step.get('args', {})
                if args.get('days') == 10:
                    print("✓ Correct forecast_days parameter passed")
                else:
                    print(f"✗ Wrong forecast_days: {args.get('days')}")
                    return False
                    
    except Exception as e:
        print(f"✗ Error with different parameters: {e}")
        return False
    
    print("\n✓ All tests passed! Orchestrator now correctly accepts user inputs.")
    return True

if __name__ == "__main__":
    success = test_orchestrator_with_inputs()
    if success:
        print("\n🎉 Chart updating issue should be fixed!")
        print("The orchestrator now uses user inputs instead of hardcoded values.")
    else:
        print("\n❌ Tests failed. There are still issues with the implementation.")
        sys.exit(1)