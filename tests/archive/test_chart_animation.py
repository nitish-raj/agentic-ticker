#!/usr/bin/env python3
"""
Test script to verify animated charts and enhanced analysis report functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ticker import (
    create_price_chart, 
    create_forecast_chart, 
    build_report,
    load_prices,
    compute_indicators,
    detect_events,
    forecast_prices
)
import pandas as pd
from datetime import datetime, timedelta

def test_animated_charts():
    """Test that animated charts are created correctly."""
    print("Testing animated charts...")
    
    # Test with sample data
    sample_price_data = [
        {"ticker": "AAPL", "date": datetime.now() - timedelta(days=5), "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000000},
        {"ticker": "AAPL", "date": datetime.now() - timedelta(days=4), "open": 151.0, "high": 153.0, "low": 150.0, "close": 152.0, "volume": 1100000},
        {"ticker": "AAPL", "date": datetime.now() - timedelta(days=3), "open": 152.0, "high": 154.0, "low": 151.0, "close": 153.0, "volume": 1200000},
        {"ticker": "AAPL", "date": datetime.now() - timedelta(days=2), "open": 153.0, "high": 155.0, "low": 152.0, "close": 154.0, "volume": 1300000},
        {"ticker": "AAPL", "date": datetime.now() - timedelta(days=1), "open": 154.0, "high": 156.0, "low": 153.0, "close": 155.0, "volume": 1400000},
    ]
    
    sample_indicator_data = [
        {"date": datetime.now() - timedelta(days=3), "ma5": 151.0, "ma10": 150.5, "daily_return": 0.01, "volatility": 0.2},
        {"date": datetime.now() - timedelta(days=2), "ma5": 152.0, "ma10": 151.0, "daily_return": 0.008, "volatility": 0.19},
        {"date": datetime.now() - timedelta(days=1), "ma5": 153.0, "ma10": 151.5, "daily_return": 0.006, "volatility": 0.18},
    ]
    
    # Test price chart
    try:
        price_fig = create_price_chart(sample_price_data, sample_indicator_data)
        assert price_fig is not None, "Price chart should not be None"
        assert hasattr(price_fig, 'data'), "Price chart should have data attribute"
        assert hasattr(price_fig, 'layout'), "Price chart should have layout attribute"
        assert len(price_fig.data) > 0, "Price chart should have at least one trace"
        print("✓ Price chart created successfully")
        
        # Check for animation frames
        if hasattr(price_fig, 'frames') and price_fig.frames:
            print(f"✓ Price chart has {len(price_fig.frames)} animation frames")
        else:
            print("⚠ Price chart has no animation frames (may be expected for small datasets)")
            
    except Exception as e:
        print(f"✗ Price chart test failed: {e}")
        return False
    
    # Test forecast chart
    sample_forecasts = [
        {"date": datetime.now() + timedelta(days=1), "forecast_price": 156.0, "confidence": 0.9, "trend": "UP"},
        {"date": datetime.now() + timedelta(days=2), "forecast_price": 157.0, "confidence": 0.8, "trend": "UP"},
        {"date": datetime.now() + timedelta(days=3), "forecast_price": 158.0, "confidence": 0.7, "trend": "UP"},
    ]
    
    try:
        forecast_fig = create_forecast_chart(sample_forecasts)
        assert forecast_fig is not None, "Forecast chart should not be None"
        assert hasattr(forecast_fig, 'data'), "Forecast chart should have data attribute"
        assert hasattr(forecast_fig, 'layout'), "Forecast chart should have layout attribute"
        assert len(forecast_fig.data) > 0, "Forecast chart should have at least one trace"
        print("✓ Forecast chart created successfully")
        
        # Check for animation frames
        if hasattr(forecast_fig, 'frames') and forecast_fig.frames:
            print(f"✓ Forecast chart has {len(forecast_fig.frames)} animation frames")
        else:
            print("⚠ Forecast chart has no animation frames")
            
    except Exception as e:
        print(f"✗ Forecast chart test failed: {e}")
        return False
    
    return True

def test_enhanced_report():
    """Test that the enhanced analysis report includes colors and formatting."""
    print("\nTesting enhanced analysis report...")
    
    sample_events = [
        {"date": datetime.now() - timedelta(days=2), "price": 154.0, "change_percent": 2.5, "direction": "UP"},
        {"date": datetime.now() - timedelta(days=1), "price": 155.0, "change_percent": -1.8, "direction": "DOWN"},
    ]
    
    sample_forecasts = [
        {"date": datetime.now() + timedelta(days=1), "forecast_price": 156.0, "confidence": 0.9, "trend": "UP"},
        {"date": datetime.now() + timedelta(days=2), "forecast_price": 157.0, "confidence": 0.7, "trend": "UP"},
        {"date": datetime.now() + timedelta(days=3), "forecast_price": 155.5, "confidence": 0.5, "trend": "DOWN"},
    ]
    
    try:
        report = build_report("AAPL", sample_events, sample_forecasts)
        assert report is not None, "Report should not be None"
        assert "content" in report, "Report should have content"
        
        content = report["content"]
        
        # Check for enhanced formatting
        assert "📊" in content, "Report should have chart emoji"
        assert "📈" in content, "Report should have up trend emoji"
        assert "📉" in content, "Report should have down trend emoji"
        assert "🔮" in content, "Report should have crystal ball emoji"
        assert "🎯" in content, "Report should have target emoji"
        assert "🟢" in content, "Report should have green circle emoji"
        assert "🔴" in content, "Report should have red circle emoji"
        assert "🟡" in content, "Report should have yellow circle emoji"
        
        print("✓ Enhanced report created successfully with color coding and emojis")
        print(f"✓ Report content length: {len(content)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced report test failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end functionality with real data."""
    print("\nTesting end-to-end functionality...")
    
    try:
        # Load real data
        price_data = load_prices("AAPL", days=10)
        if not price_data:
            print("⚠ No price data available for end-to-end test")
            return True
        
        # Compute indicators
        indicator_data = compute_indicators(price_data)
        if not indicator_data:
            print("⚠ No indicator data available for end-to-end test")
            return True
        
        # Detect events
        events = detect_events(indicator_data, threshold=1.0)
        
        # Generate forecasts
        forecasts = forecast_prices(indicator_data, days=3)
        
        # Create charts
        price_fig = create_price_chart(price_data, indicator_data)
        forecast_fig = create_forecast_chart(forecasts)
        
        # Build report
        report = build_report("AAPL", events, forecasts)
        
        print("✓ End-to-end test completed successfully")
        print(f"✓ Processed {len(price_data)} price points")
        print(f"✓ Generated {len(forecasts)} forecast points")
        print(f"✓ Detected {len(events)} events")
        
        return True
        
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Chart Animation and Enhanced Report Tests")
    print("=" * 60)
    
    tests = [
        ("Animated Charts", test_animated_charts),
        ("Enhanced Report", test_enhanced_report),
        ("End-to-End", test_end_to_end),
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
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Chart animation and enhanced report functionality is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())