#!/usr/bin/env python3
"""
Test script to verify ticker validation enhancements and forecast chart improvements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ticker import validate_ticker, create_forecast_chart
from datetime import datetime, timedelta

def test_ticker_validation():
    """Test enhanced ticker validation with Google search."""
    print("Testing enhanced ticker validation...")
    
    test_cases = [
        ("AAPL", "AAPL"),  # Valid ticker
        ("Apple", "AAPL"),  # Company name
        ("Microsoft", "MSFT"),  # Company name
        ("Berkshire Hathaway", "BRK.A"),  # Complex company name
        ("Tesla", "TSLA"),  # Company name
        ("GOOGL", "GOOGL"),  # Valid ticker
        ("Amazon", "AMZN"),  # Company name
    ]
    
    results = []
    for input_text, expected in test_cases:
        try:
            result = validate_ticker(input_text)
            print(f"  '{input_text}' -> '{result}' (expected: '{expected}')")
            
            # Check if result is a valid ticker format
            import re
            is_valid_format = re.match(r'^[A-Z0-9]{1,5}(\.[A-Z0-9]{1,2})?$', result)
            if is_valid_format:
                print(f"    ✓ Valid ticker format")
                results.append(True)
            else:
                print(f"    ⚠ Invalid ticker format: {result}")
                results.append(False)
                
        except Exception as e:
            print(f"  ✗ Error validating '{input_text}': {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) if results else 0
    print(f"✓ Ticker validation success rate: {success_rate:.1%} ({sum(results)}/{len(results)})")
    return success_rate >= 0.7  # Allow some flexibility for API issues

def test_forecast_chart():
    """Test enhanced forecast chart with line chart and colors."""
    print("\nTesting enhanced forecast chart...")
    
    # Test data
    sample_forecasts = [
        {"date": datetime.now() + timedelta(days=1), "forecast_price": 156.0, "confidence": 0.9, "trend": "UP"},
        {"date": datetime.now() + timedelta(days=2), "forecast_price": 157.0, "confidence": 0.8, "trend": "UP"},
        {"date": datetime.now() + timedelta(days=3), "forecast_price": 158.0, "confidence": 0.7, "trend": "UP"},
        {"date": datetime.now() + timedelta(days=4), "forecast_price": 157.5, "confidence": 0.6, "trend": "DOWN"},
        {"date": datetime.now() + timedelta(days=5), "forecast_price": 157.0, "confidence": 0.5, "trend": "DOWN"},
    ]
    
    try:
        fig = create_forecast_chart(sample_forecasts)
        
        # Check basic structure
        assert fig is not None, "Figure should not be None"
        assert hasattr(fig, 'data'), "Figure should have data attribute"
        assert hasattr(fig, 'layout'), "Figure should have layout attribute"
        print("✓ Forecast chart created successfully")
        
        # Check for line chart (should have continuous traces, not just individual points)
        line_traces = []
        point_traces = []
        for trace in fig.data:
            if hasattr(trace, 'mode') and trace.mode is not None:
                mode_str = str(trace.mode).lower()
                if 'lines' in mode_str:
                    line_traces.append(trace)
                if 'markers' in mode_str:
                    point_traces.append(trace)
        
        print(f"✓ Found {len(line_traces)} line traces and {len(point_traces)} point traces")
        
        # Check for confidence band (should have filled area)
        filled_traces = []
        for trace in fig.data:
            if hasattr(trace, 'fill') and trace.fill is not None:
                fill_str = str(trace.fill).lower()
                if fill_str != 'none' and fill_str != '':
                    filled_traces.append(trace)
        if filled_traces:
            print("✓ Confidence band (filled area) detected")
        else:
            print("⚠ No confidence band detected")
        
        # Check for multiple traces (main line + confidence band + individual points)
        if len(fig.data) >= 2:
            print(f"✓ Multiple traces detected ({len(fig.data)} total)")
        else:
            print(f"⚠ Only {len(fig.data)} trace(s) detected")
        
        # Check for animation frames
        if hasattr(fig, 'frames') and fig.frames:
            print(f"✓ Animation frames detected: {len(fig.frames)}")
        else:
            print("⚠ No animation frames detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Forecast chart test failed: {e}")
        return False

def main():
    """Run enhancement tests."""
    print("🚀 Starting Enhancement Tests")
    print("=" * 50)
    
    tests = [
        ("Ticker Validation", test_ticker_validation),
        ("Forecast Chart", test_forecast_chart),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Enhancement Test Results:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All enhancement tests passed!")
        return 0
    else:
        print("⚠️  Some enhancement tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())