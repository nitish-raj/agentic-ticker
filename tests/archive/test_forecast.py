#!/usr/bin/env python3
"""
Test the forecasting functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ticker import load_prices, compute_indicators, forecast_prices

def test_forecast():
    """Test the forecasting functionality"""
    print("Testing forecasting functionality...")
    
    # Load price data
    print("Loading price data...")
    price_data = load_prices("AAPL", 30)
    print(f"Loaded {len(price_data)} price records")
    
    # Compute indicators
    print("Computing indicators...")
    indicator_data = compute_indicators(price_data)
    print(f"Computed {len(indicator_data)} indicator records")
    
    # Forecast prices
    print("Forecasting prices...")
    forecasts = forecast_prices(indicator_data, 5)
    print(f"Generated {len(forecasts)} forecasts")
    
    if forecasts:
        print("Sample forecasts:")
        for i, forecast in enumerate(forecasts[:3]):
            print(f"  {forecast['date'].strftime('%Y-%m-%d')}: ${forecast['forecast_price']:.2f} (confidence: {forecast['confidence']:.2f})")
    
    return True

if __name__ == "__main__":
    success = test_forecast()
    sys.exit(0 if success else 1)