"""
Data formatting utilities for display in Streamlit application.
Provides consistent formatting for all data types including column names,
percentages, and decimal rounding.
"""

import pandas as pd
from typing import List, Dict, Any, Union


def to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(word.capitalize() for word in components[1:])


def format_percentage(value: Union[float, int]) -> str:
    """Format a value as a percentage with 2 decimal places."""
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return "0.00%"


def format_decimal(value: Union[float, int], decimal_places: int = 2) -> str:
    """Format a number with specified decimal places."""
    try:
        return f"{float(value):.{decimal_places}f}"
    except (ValueError, TypeError):
        return f"0.{decimal_places}f"


def format_dataframe_for_display(
    data: Union[List[Dict[str, Any]], pd.DataFrame], data_type: str = "general"
) -> pd.DataFrame:
    """
    Format dataframe for display with camel case columns, percentages, and rounded decimals.

    Args:
        data: List of dictionaries or pandas DataFrame
        data_type: Type of data ("price", "indicator", "events", "forecasts", "general")

    Returns:
        Formatted pandas DataFrame
    """
    # Convert to DataFrame if needed
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Convert column names to camel case
    df.columns = [to_camel_case(col) for col in df.columns]

    # Apply specific formatting based on data type
    if data_type == "price":
        # Price data formatting
        numeric_columns = ["open", "high", "low", "close"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: format_decimal(x, 2))

        # Volume formatting (no decimal places)
        if "volume" in df.columns:
            df["volume"] = df["volume"].apply(lambda x: f"{int(x):,}")

    elif data_type == "indicator":
        # Indicator data formatting
        if "dailyReturn" in df.columns:
            df["dailyReturn"] = df["dailyReturn"].apply(format_percentage)

        if "volatility" in df.columns:
            df["volatility"] = df["volatility"].apply(format_percentage)

        # Moving averages with 2 decimal places
        for col in ["ma5", "ma10"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: format_decimal(x, 2))

    elif data_type == "events":
        # Events data formatting
        if "changePercent" in df.columns:
            df["changePercent"] = df["changePercent"].apply(format_percentage)

        if "price" in df.columns:
            df["price"] = df["price"].apply(lambda x: format_decimal(x, 2))

    elif data_type == "forecasts":
        # Forecasts data formatting
        if "forecastPrice" in df.columns:
            df["forecastPrice"] = df["forecastPrice"].apply(
                lambda x: format_decimal(x, 2)
            )

        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].apply(format_percentage)

    else:
        # General formatting - round all numeric columns to 2 decimal places
        for col in df.select_dtypes(include=["number"]).columns:
            if col not in ["volume"]:  # Keep volume as integer
                df[col] = df[col].apply(lambda x: format_decimal(x, 2))

    return df


def format_price_data(data: Union[List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
    """Format price data for display."""
    return format_dataframe_for_display(data, "price")


def format_indicator_data(
    data: Union[List[Dict[str, Any]], pd.DataFrame],
) -> pd.DataFrame:
    """Format indicator data for display."""
    return format_dataframe_for_display(data, "indicator")


def format_events_data(data: Union[List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
    """Format events data for display."""
    return format_dataframe_for_display(data, "events")


def format_forecasts_data(
    data: Union[List[Dict[str, Any]], pd.DataFrame],
) -> pd.DataFrame:
    """Format forecasts data for display."""
    return format_dataframe_for_display(data, "forecasts")
