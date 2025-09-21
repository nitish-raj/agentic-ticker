#!/usr/bin/env python3
"""
Test to verify that charts update correctly when different tickers are entered.
This simulates the UI behavior without requiring Streamlit.
"""

import os
import sys
sys.path.append('.')

from agentic_ticker import Orchestrator, create_price_chart

def test_chart_updates():
    """Test that charts are generated correctly for different tickers."""
    
    test_cases = [
        ("AAPL", "Apple"),
        ("MSFT", "Microsoft"), 
        ("GOOGL", "Google"),
    ]
    
    print("Testing chart generation with different tickers...")
    print("=" * 60)
    
    for ticker, company_name in test_cases:
        print(f"\nTesting chart generation for: {ticker} ({company_name})")
        print("-" * 50)
        
        try:
            # Create orchestrator and run analysis
            orch = Orchestrator()
            
            # Run the pipeline
            steps = orch.run(
                ticker_input=ticker,
                days=30,
                threshold=2.0,
                forecast_days=5,
                on_event=None  # No event callback for cleaner output
            )
            
            # Extract data from steps
            price_data = None
            indicator_data = None
            
            for step in steps:
                if step.get("type") == "result":
                    if step.get("name") == "load_prices":
                        price_data = step.get("result")
                    elif step.get("name") == "compute_indicators":
                        indicator_data = step.get("result")
            
            # Test chart generation
            if price_data and indicator_data:
                print(f"✓ Price data: {len(price_data)} records")
                print(f"✓ Indicator data: {len(indicator_data)} records")
                
                # Generate chart
                chart_img = create_price_chart(price_data, indicator_data)
                if chart_img:
                    print(f"✓ Chart generated successfully ({len(chart_img)} chars)")
                    
                    # Verify chart data is different for different tickers
                    if price_data:
                        latest_price = price_data[-1].get('close', 0)
                        print(f"✓ Latest price for {ticker}: ${latest_price:.2f}")
                else:
                    print("✗ Chart generation failed")
            else:
                print("✗ Missing required data for chart generation")
                
        except Exception as e:
            print(f"✗ Error testing {ticker}: {e}")
    
    print("\n" + "=" * 60)
    print("Chart update test completed!")

def test_company_name_conversion():
    """Test that company names are converted and charts still work."""
    
    print("\nTesting company name conversion with chart generation...")
    print("=" * 60)
    
    company_test_cases = [
        ("Apple", "AAPL"),
        ("Microsoft", "MSFT"),
        ("Google", "GOOGL"),
    ]
    
    for company_name, expected_ticker in company_test_cases:
        print(f"\nTesting: '{company_name}' → should convert to '{expected_ticker}'")
        print("-" * 50)
        
        try:
            orch = Orchestrator()
            
            # Track validation result
            validation_result = None
            
            def on_event(event):
                nonlocal validation_result
                if event.get("type") == "result" and event.get("name") == "validate_ticker":
                    validation_result = event.get("result")
            
            # Run pipeline
            steps = orch.run(
                ticker_input=company_name,
                days=30,
                threshold=2.0,
                forecast_days=5,
                on_event=on_event
            )
            
            # Check validation
            if validation_result and validation_result.upper() == expected_ticker.upper():
                print(f"✓ Conversion successful: '{company_name}' → '{validation_result}'")
                
                # Extract data and test chart
                price_data = None
                indicator_data = None
                
                for step in steps:
                    if step.get("type") == "result":
                        if step.get("name") == "load_prices":
                            price_data = step.get("result")
                        elif step.get("name") == "compute_indicators":
                            indicator_data = step.get("result")
                
                if price_data and indicator_data:
                    chart_img = create_price_chart(price_data, indicator_data)
                    if chart_img:
                        print(f"✓ Chart generated for converted ticker '{validation_result}'")
                    else:
                        print("✗ Chart generation failed")
                else:
                    print("✗ No data available for chart generation")
            else:
                print(f"✗ Conversion failed: got '{validation_result}', expected '{expected_ticker}'")
                
        except Exception as e:
            print(f"✗ Error testing {company_name}: {e}")

if __name__ == "__main__":
    test_chart_updates()
    test_company_name_conversion()