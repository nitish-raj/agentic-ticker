#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from agentic_ticker import create_price_chart

def test_price_animation():
    """Test price chart animation consistency."""
    print("📈 Testing price chart animation...")
    
    # Create sample price data
    base_date = datetime.now() - timedelta(days=30)
    price_data = []
    for i in range(30):
        date = base_date + timedelta(days=i)
        close_price = 150 + i * 0.5 + (i % 5) * 2  # Some variation
        price_data.append({
            'date': date,
            'open': close_price - 1,
            'high': close_price + 2,
            'low': close_price - 2,
            'close': close_price,
            'volume': 1000000 + i * 50000
        })
    
    # Create sample indicator data
    indicator_data = []
    for i in range(30):
        date = base_date + timedelta(days=i)
        close_price = 150 + i * 0.5 + (i % 5) * 2
        indicator_data.append({
            'date': date,
            'ma5': close_price + 0.3,  # Simplified MA
            'ma10': close_price - 0.2,  # Simplified MA
            'rsi': 50 + (i % 10),
            'macd': 0.1 + (i % 5) * 0.1
        })
    
    try:
        fig = create_price_chart(price_data, indicator_data)
        print("✓ Price chart created successfully")
        
        # Check initial chart structure
        print(f"Initial chart has {len(fig.data)} traces:")
        for i, trace in enumerate(fig.data):
            name = getattr(trace, 'name', 'Unknown')
            mode = getattr(trace, 'mode', 'None')
            print(f"  Trace {i}: {name} (mode={mode})")
        
        # Check animation frames
        if hasattr(fig, 'frames') and fig.frames:
            print(f"\nAnimation has {len(fig.frames)} frames:")
            
            # Check first few frames
            for i in range(min(3, len(fig.frames))):
                frame = fig.frames[i]
                print(f"  Frame {i} has {len(frame.data)} traces:")
                for j, trace in enumerate(frame.data):
                    name = getattr(trace, 'name', 'Unknown')
                    mode = getattr(trace, 'mode', 'None')
                    x_len = len(getattr(trace, 'x', []))
                    print(f"    Trace {j}: {name} (mode={mode}, points={x_len})")
            
            # Check last frame
            if len(fig.frames) > 0:
                last_frame = fig.frames[-1]
                print(f"  Frame {len(fig.frames)-1} (last) has {len(last_frame.data)} traces:")
                for j, trace in enumerate(last_frame.data):
                    name = getattr(trace, 'name', 'Unknown')
                    mode = getattr(trace, 'mode', 'None')
                    x_len = len(getattr(trace, 'x', []))
                    print(f"    Trace {j}: {name} (mode={mode}, points={x_len})")
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
        
        print("✓ Price animation test completed")
        return True
        
    except Exception as e:
        print(f"✗ Price animation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_price_animation()