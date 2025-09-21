#!/usr/bin/env python3
"""
Debug the indicator computation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from agentic_ticker import load_prices

def debug_indicators():
    """Debug the indicator computation"""
    print("Debugging indicator computation...")
    
    # Load price data
    print("Loading price data...")
    price_data = load_prices("AMZN", 30)
    print(f"Loaded {len(price_data)} price records")
    
    if not price_data:
        print("No price data loaded")
        return
    
    print("First few records:")
    for i, record in enumerate(price_data[:3]):
        print(f"  {i+1}: {record}")
    
    # Debug compute_indicators logic
    print("\nDebugging compute_indicators logic...")
    df = pd.DataFrame(price_data).sort_values('date')
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    df['daily_return'] = df['close'].pct_change()
    print(f"After daily_return: {df.shape}")
    print(f"NaN in daily_return: {df['daily_return'].isna().sum()}")
    
    df['ma5'] = df['close'].rolling(5).mean()
    print(f"After MA5: {df.shape}")
    print(f"NaN in MA5: {df['ma5'].isna().sum()}")
    
    df['ma20'] = df['close'].rolling(20).mean()
    print(f"After MA20: {df.shape}")
    print(f"NaN in MA20: {df['ma20'].isna().sum()}")
    
    df['annualized_vol'] = df['daily_return'].rolling(30).std() * np.sqrt(252)
    print(f"After annualized_vol: {df.shape}")
    print(f"NaN in annualized_vol: {df['annualized_vol'].isna().sum()}")
    
    print(f"Total NaN rows: {df.isna().sum().sum()}")
    
    # Drop rows with NaN values
    df_clean = df.dropna(subset=['ma5', 'ma20', 'daily_return', 'annualized_vol'])
    print(f"After dropping NaN: {df_clean.shape}")
    
    if df_clean.empty:
        print("No data left after dropping NaN values")
        print("This is likely because we need more data for the longer moving averages")
        print("Let's try with a smaller window for volatility...")
        
        # Try with smaller window
        df['annualized_vol_small'] = df['daily_return'].rolling(5).std() * np.sqrt(252)
        print(f"Small window volatility NaN: {df['annualized_vol_small'].isna().sum()}")
        
        df_clean_small = df.dropna(subset=['ma5', 'daily_return', 'annualized_vol_small'])
        print(f"Small window clean data: {df_clean_small.shape}")
        
        if not df_clean_small.empty:
            print("Small window approach works!")

if __name__ == "__main__":
    debug_indicators()