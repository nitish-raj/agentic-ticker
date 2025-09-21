#!/usr/bin/env python3
"""
Test the enhanced JSON display functionality in the agent loop.
"""

import sys
import os
sys.path.append('.')

from agentic_ticker import Orchestrator, _format_json_for_display

def test_json_display_enhancement():
    """Test that JSON arguments are displayed in a readable format."""
    print("Testing enhanced JSON display functionality...")
    print("=" * 50)
    
    # Test the formatting function directly
    test_args = {
        'ticker': 'AAPL',
        'days': 30,
        'threshold': 2.0,
        'price_data': [
            {'date': '2025-09-21', 'close': 150.25, 'volume': 1000000},
            {'date': '2025-09-20', 'close': 148.50, 'volume': 950000}
        ]
    }
    
    print("Test arguments:")
    print(test_args)
    print("\nFormatted output:")
    formatted = _format_json_for_display(test_args)
    print(formatted)
    
    # Verify the formatting
    assert '{\n  ' in formatted, "JSON should be formatted with indentation"
    assert '"ticker": "AAPL"' in formatted, "Should contain ticker info"
    assert '"days": 30' in formatted, "Should contain days info"
    assert '"threshold": 2.0' in formatted, "Should contain threshold info"
    assert '"price_data"' in formatted, "Should contain price_data array"
    
    print("\n✅ JSON formatting test passed!")
    
    # Test with a simple orchestrator run to capture events
    print("\nTesting agent loop with enhanced display...")
    events_captured = []
    
    def capture_event(event):
        events_captured.append(event)
        kind = event.get("type")
        if kind == "call":
            args = event.get('args', {})
            formatted_args = _format_json_for_display(args)
            print(f"Event captured: Step {event.get('step')}: {event.get('name')}")
            print(f"Formatted args preview: {formatted_args[:100]}...")
    
    try:
        orch = Orchestrator()
        # Run a quick analysis
        steps = orch.run('AAPL', 5, 2.0, 3, on_event=capture_event)
        
        # Check that we captured call events
        call_events = [e for e in events_captured if e.get('type') == 'call']
        assert len(call_events) > 0, "Should have captured call events"
        
        print(f"\n✅ Captured {len(call_events)} call events with enhanced display")
        
        # Show a sample of the enhanced display
        if call_events:
            sample_event = call_events[0]
            args = sample_event.get('args', {})
            formatted_args = _format_json_for_display(args)
            print(f"\nSample enhanced display for {sample_event.get('name')}:")
            print("-" * 40)
            print(formatted_args)
            print("-" * 40)
        
        print("\n✅ Enhanced JSON display test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_json_display_enhancement()
    if success:
        print("\n🎉 All tests passed! Enhanced JSON display is working correctly.")
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        sys.exit(1)