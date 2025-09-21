#!/usr/bin/env python3
"""
Test script to check data types and structures
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ticker import load_prices, compute_indicators
import pandas as pd

def test_data_types():
    """Test data types and structures"""
    print("Loading price data...")
    price_data = load_prices("AAPL", days=30)
    print(f"Loaded {len(price_data)} items")
    
    if not price_data:
        print("No data loaded")
        return True
    
    print("Checking first item structure...")
    first_item = price_data[0]
    print(f"Type: {type(first_item)}")
    print(f"Keys: {list(first_item.keys())}")
    for key, value in first_item.items():
        print(f"  {key}: {type(value)} = {value}")
    
    print("\nTrying to create DataFrame directly...")
    try:
        df = pd.DataFrame(price_data)
        print(f"DataFrame created successfully with shape {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Failed to create DataFrame: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nCalling compute_indicators...")
    try:
        result = compute_indicators(price_data)
        print(f"compute_indicators returned {len(result)} items")
    except Exception as e:
        print(f"compute_indicators failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_data_types()
    sys.exit(0 if success else 1)