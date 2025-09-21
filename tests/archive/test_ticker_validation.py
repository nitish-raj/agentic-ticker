#!/usr/bin/env python3
"""
Test script to verify the ticker validation feature works correctly.
Tests both ticker symbols and company names.
"""

import os
import sys
sys.path.append('.')

from agentic_ticker import validate_ticker

def test_ticker_validation():
    """Test the validate_ticker function with various inputs."""
    
    test_cases = [
        # Test valid ticker symbols (should return unchanged)
        ("AAPL", "AAPL"),
        ("MSFT", "MSFT"), 
        ("GOOGL", "GOOGL"),
        ("aapl", "AAPL"),  # Test lowercase conversion
        
        # Test company names (should be converted to tickers)
        ("Apple", "AAPL"),
        ("Microsoft", "MSFT"),
        ("Google", "GOOGL"),
        
        # Test edge cases
        ("INVALID", "INVALID"),  # Should fallback to uppercase if invalid
        ("", ""),  # Empty string
    ]
    
    print("Testing ticker validation...")
    print("=" * 50)
    
    for input_text, expected in test_cases:
        try:
            result = validate_ticker(input_text)
            status = "✓" if result.upper() == expected.upper() else "✗"
            print(f"{status} Input: '{input_text}' -> Result: '{result}' (Expected: '{expected}')")
        except Exception as e:
            print(f"✗ Input: '{input_text}' -> Error: {e}")
    
    print("\nTesting with actual yfinance validation...")
    print("=" * 50)
    
    # Test with real validation
    real_test_cases = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    for ticker in real_test_cases:
        try:
            result = validate_ticker(ticker)
            print(f"✓ '{ticker}' -> '{result}' (validated successfully)")
        except Exception as e:
            print(f"✗ '{ticker}' -> Error: {e}")

if __name__ == "__main__":
    test_ticker_validation()