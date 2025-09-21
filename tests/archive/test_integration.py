#!/usr/bin/env python3
"""
Integration test to verify the full pipeline works with ticker validation.
"""

import os
import sys
sys.path.append('.')

from agentic_ticker import Orchestrator

def test_integration():
    """Test the full pipeline with different inputs."""
    
    test_cases = [
        # Test with ticker symbol
        ("AAPL", "AAPL"),
        # Test with company name
        ("Microsoft", "MSFT"),
        # Test with lowercase ticker
        ("googl", "GOOGL"),
    ]
    
    print("Testing full integration pipeline...")
    print("=" * 60)
    
    for input_text, expected_ticker in test_cases:
        print(f"\nTesting with input: '{input_text}'")
        print("-" * 40)
        
        try:
            # Create orchestrator
            orch = Orchestrator()
            
            # Define event callback to capture progress
            events_log = []
            
            def on_event(event):
                events_log.append(event)
                if event.get("type") == "info":
                    print(f"INFO: {event.get('message')}")
                elif event.get("type") == "result":
                    step = event.get("step")
                    name = event.get("name")
                    result = event.get("result")
                    if name == "validate_ticker":
                        print(f"Step {step}: Validated ticker -> '{result}'")
                    elif name == "load_prices":
                        print(f"Step {step}: Loaded {len(result)} price records")
                    elif name == "compute_indicators":
                        print(f"Step {step}: Computed {len(result)} indicator records")
                    elif name == "detect_events":
                        print(f"Step {step}: Detected {len(result)} events")
                    elif name == "forecast_prices":
                        print(f"Step {step}: Generated {len(result)} forecasts")
                    elif name == "build_report":
                        print(f"Step {step}: Built report for {result.get('ticker')}")
                elif event.get("type") == "error":
                    print(f"ERROR: {event.get('error')}")
            
            # Run the pipeline
            steps = orch.run(
                ticker_input=input_text,
                days=10,  # Shorter period for faster testing
                threshold=5.0,  # Higher threshold for fewer events
                forecast_days=3,  # Shorter forecast
                on_event=on_event
            )
            
            # Check if validation worked correctly
            validation_step = None
            for step in steps:
                if step.get("type") == "result" and step.get("name") == "validate_ticker":
                    validation_step = step
                    break
            
            if validation_step:
                validated_ticker = validation_step.get("result")
                if validated_ticker and validated_ticker.upper() == expected_ticker.upper():
                    print(f"✓ SUCCESS: '{input_text}' correctly validated to '{validated_ticker}'")
                else:
                    print(f"✗ FAILED: '{input_text}' -> '{validated_ticker}' (expected '{expected_ticker}')")
            else:
                print(f"✗ FAILED: No validation step found for '{input_text}'")
                
        except Exception as e:
            print(f"✗ ERROR: Failed to process '{input_text}': {e}")
    
    print("\n" + "=" * 60)
    print("Integration test completed!")

if __name__ == "__main__":
    test_integration()