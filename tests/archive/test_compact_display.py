#!/usr/bin/env python3
"""
Test the compact JSON display with truncation and better space management.
"""

import sys
import os
sys.path.append('.')

from agentic_ticker import Orchestrator, _truncate_large_data, _format_json_for_display

def test_compact_display():
    """Test that JSON display is compact and handles large data well."""
    print("Testing compact JSON display with truncation...")
    print("=" * 60)
    
    # Test with data that would normally be very long
    large_args = {
        'ticker': 'AAPL',
        'days': 365,  # Large number
        'threshold': 2.0,
        'price_data': [
            {'date': f'2025-09-{i:02d}', 'close': 150.0 + i, 'volume': 1000000 + i*10000}
            for i in range(1, 31)  # 30 items - should be truncated
        ],
        'very_long_description': 'This is an extremely long description that would normally take up a lot of space in the display and make it hard to read for users who want to quickly understand what arguments are being passed to the function without having to scroll through massive amounts of data'
    }
    
    print("Original data size:")
    print(f"- price_data items: {len(large_args['price_data'])}")
    print(f"- description length: {len(large_args['very_long_description'])} chars")
    
    # Test truncation
    truncated = _truncate_large_data(large_args)
    print(f"\nAfter truncation:")
    print(f"- price_data items: {len(truncated['price_data'])}")
    print(f"- description length: {len(truncated['very_long_description'])} chars")
    print(f"- Last price_data item: {truncated['price_data'][-1]}")
    
    # Test formatting
    formatted = _format_json_for_display(large_args)
    print(f"\nFormatted output length: {len(formatted)} chars")
    print("Formatted output preview:")
    print("-" * 40)
    print(formatted[:300] + "..." if len(formatted) > 300 else formatted)
    print("-" * 40)
    
    # Verify truncation worked
    assert len(truncated['price_data']) <= 4, "Price data should be truncated to 3 items + 1 summary"
    assert any('more items' in str(item) for item in truncated['price_data']), "Should indicate truncation"
    assert len(truncated['very_long_description']) <= 53, "Long strings should be truncated"
    assert truncated['very_long_description'].endswith('...'), "Truncated strings should end with ..."
    
    print("\n✅ Truncation test passed!")
    
    # Test with actual orchestrator to see display in action
    print("\nTesting agent loop with compact display...")
    events_captured = []
    
    def capture_event(event):
        events_captured.append(event)
        kind = event.get("type")
        if kind == "call":
            args = event.get('args', {})
            formatted_args = _format_json_for_display(args)
            print(f"Step {event.get('step')}: {event.get('name')}")
            print(f"Args length: {len(formatted_args)} chars")
            # Show first few lines
            lines = formatted_args.split('\n')
            preview = '\n'.join(lines[:8]) + ('\n...' if len(lines) > 8 else '')
            print(f"Preview:\n{preview}\n")
    
    try:
        orch = Orchestrator()
        # Run with longer period to generate more data
        steps = orch.run('AAPL', 60, 2.0, 5, on_event=capture_event)
        
        call_events = [e for e in events_captured if e.get('type') == 'call']
        print(f"\n✅ Captured {len(call_events)} call events with compact display")
        
        # Check that displays are reasonably sized
        for event in call_events:
            args = event.get('args', {})
            formatted_args = _format_json_for_display(args)
            # Should be much more compact now
            if len(formatted_args) > 1000:  # Still reasonable limit
                print(f"Warning: {event.get('name')} args still long: {len(formatted_args)} chars")
        
        print("\n✅ Compact display test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_compact_display()
    if success:
        print("\n🎉 All tests passed! Compact JSON display is working correctly.")
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        sys.exit(1)