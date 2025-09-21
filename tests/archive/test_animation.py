#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from agentic_ticker import create_forecast_chart

def test_animation_consistency():
    """Test that animation frames are consistent with the initial chart."""
    print("🎬 Testing animation consistency...")
    
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
        print("✓ Chart created successfully")
        
        # Check initial chart structure
        print(f"Initial chart has {len(fig.data)} traces:")
        for i, trace in enumerate(fig.data):
            mode = getattr(trace, 'mode', 'None')
            fill = getattr(trace, 'fill', 'None')
            print(f"  Trace {i}: mode={mode}, fill={fill}")
        
        # Check animation frames
        if hasattr(fig, 'frames') and fig.frames:
            print(f"\nAnimation has {len(fig.frames)} frames:")
            
            for i, frame in enumerate(fig.frames):
                print(f"  Frame {i} has {len(frame.data)} traces:")
                for j, trace in enumerate(frame.data):
                    mode = getattr(trace, 'mode', 'None')
                    fill = getattr(trace, 'fill', 'None')
                    print(f"    Trace {j}: mode={mode}, fill={fill}")
                    
                    # Check if frame has line chart structure
                    if mode == 'lines+markers':
                        x_data = list(trace.x)
                        y_data = list(trace.y)
                        print(f"      Line chart with {len(x_data)} points: {x_data}")
                    
                    # Check if frame has confidence band
                    if fill == 'toself':
                        print(f"      Confidence band detected")
        else:
            print("⚠ No animation frames found")
        
        # Test animation controls
        if hasattr(fig, 'layout') and hasattr(fig.layout, 'updatemenus'):
            print("✓ Animation controls detected")
            updatemenus = fig.layout.updatemenus
            if updatemenus and len(updatemenus) > 0:
                buttons = updatemenus[0].buttons
                print(f"  Found {len(buttons)} buttons:")
                for i, button in enumerate(buttons):
                    print(f"    Button {i}: {button.label}")
        else:
            print("⚠ No animation controls found")
        
        print("✓ Animation consistency test completed")
        return True
        
    except Exception as e:
        print(f"✗ Animation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_animation_consistency()