#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from agentic_ticker import create_forecast_chart

def debug_forecast_chart():
    """Debug the forecast chart creation."""
    print("🔍 Debugging forecast chart creation...")
    
    # Test data
    sample_forecasts = [
        {"date": datetime.now() + timedelta(days=1), "forecast_price": 156.0, "confidence": 0.9, "trend": "UP"},
        {"date": datetime.now() + timedelta(days=2), "forecast_price": 157.0, "confidence": 0.8, "trend": "UP"},
        {"date": datetime.now() + timedelta(days=3), "forecast_price": 158.0, "confidence": 0.7, "trend": "UP"},
        {"date": datetime.now() + timedelta(days=4), "forecast_price": 157.5, "confidence": 0.6, "trend": "DOWN"},
        {"date": datetime.now() + timedelta(days=5), "forecast_price": 157.0, "confidence": 0.5, "trend": "DOWN"},
    ]
    
    try:
        print("Creating forecast chart...")
        fig = create_forecast_chart(sample_forecasts)
        print("✓ Chart created successfully")
        
        print("Checking figure attributes...")
        print(f"  - Figure type: {type(fig)}")
        print(f"  - Has data: {hasattr(fig, 'data')}")
        print(f"  - Has layout: {hasattr(fig, 'layout')}")
        
        if hasattr(fig, 'data'):
            print(f"  - Data length: {len(fig.data)}")
            print(f"  - Data type: {type(fig.data)}")
            
            # Check each trace
            for i, trace in enumerate(fig.data):
                print(f"  - Trace {i}: {type(trace)}")
                if hasattr(trace, 'mode'):
                    print(f"    - Mode: {trace.mode}")
                if hasattr(trace, 'fill'):
                    print(f"    - Fill: {trace.fill}")
        
        print("✓ All checks passed")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_forecast_chart()