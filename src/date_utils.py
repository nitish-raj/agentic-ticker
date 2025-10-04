import pandas as pd
from datetime import datetime
from typing import Optional, Union, List, Any
import logging

logger = logging.getLogger(__name__)


def format_datetime_as_date(date_input: Any) -> str:
    """
    Convert datetime object or string to date-only string format (YYYY-MM-DD).

    Args:
        date_input: datetime object, pandas Timestamp, or date string

    Returns:
        Date string in YYYY-MM-DD format, or original input if conversion fails
    """
    try:
        # Handle None or empty input
        if date_input is None or date_input == "":
            return ""

        # If already a string, try to parse and reformat
        if isinstance(date_input, str):
            # Try to parse as datetime first
            try:
                dt = pd.to_datetime(date_input)
                return dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                # If parsing fails, return original string
                return date_input

        # Handle pandas Timestamp
        if hasattr(date_input, "to_pydatetime"):
            dt = date_input.to_pydatetime()
            return dt.strftime("%Y-%m-%d")

        # Handle datetime object
        if isinstance(date_input, datetime):
            return date_input.strftime("%Y-%m-%d")

        # Handle other types (try to convert to datetime)
        dt = pd.to_datetime(date_input)
        return dt.strftime("%Y-%m-%d")

    except Exception as e:
        logger.warning(f"Failed to format date {date_input}: {e}")
        return str(date_input) if date_input is not None else ""


def safe_to_datetime(
    series: Union[pd.Series, pd.DataFrame],
    errors: str = "coerce",
    date_format: Optional[str] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Safely convert series to datetime with error handling."""
    try:
        if date_format:
            return pd.to_datetime(series, format=date_format, errors=errors)  # type: ignore
        else:
            return pd.to_datetime(series, errors=errors)  # type: ignore
    except Exception as e:
        logger.error(f"Error converting to datetime: {str(e)}")
        if errors == "coerce":
            return pd.to_datetime(pd.Series([None] * len(series)), errors="coerce")  # type: ignore
        raise


def sort_by_date(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Sort DataFrame by date column."""
    if df.empty:
        return df

    if date_column not in df.columns:
        logger.warning(f"Date column '{date_column}' not found in DataFrame")
        return df

    try:
        # Ensure the date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = safe_to_datetime(df[date_column])

        # Sort by date
        return df.sort_values(date_column)
    except Exception as e:
        logger.error(f"Error sorting by date: {str(e)}")
        return df


def validate_date_column(df: pd.DataFrame, date_column: str = "date") -> bool:
    """Validate that date column exists and is proper format."""
    if df.empty:
        return True

    if date_column not in df.columns:
        logger.error(f"Date column '{date_column}' not found in DataFrame")
        return False

    try:
        # Try to convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            temp_series = safe_to_datetime(df[date_column])
            if isinstance(temp_series, pd.Series) and temp_series.isna().all():
                logger.error(f"Date column '{date_column}' contains no valid dates")
                return False
    except Exception as e:
        logger.error(f"Error validating date column '{date_column}': {str(e)}")
        return False

    return True


def get_date_range(df: pd.DataFrame, date_column: str = "date") -> Optional[tuple]:
    """Get the date range (min, max) from a DataFrame."""
    if df.empty or date_column not in df.columns:
        return None

    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = safe_to_datetime(df[date_column])

        valid_dates = df[date_column].dropna()
        if valid_dates.empty:
            return None

        return (valid_dates.min(), valid_dates.max())
    except Exception as e:
        logger.error(f"Error getting date range: {str(e)}")
        return None


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    date_column: str = "date",
) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    if df.empty:
        return df

    if date_column not in df.columns:
        logger.warning(f"Date column '{date_column}' not found in DataFrame")
        return df

    try:
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = safe_to_datetime(df[date_column])

        filtered_df = df.copy()

        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df[date_column] >= start_date]

        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df[date_column] <= end_date]

        return filtered_df if isinstance(filtered_df, pd.DataFrame) else df
    except Exception as e:
        logger.error(f"Error filtering by date range: {str(e)}")
        return df


def add_date_components(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Add date components (year, month, day, dayofweek) to DataFrame."""
    if df.empty or date_column not in df.columns:
        return df

    try:
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = safe_to_datetime(df[date_column])

        df = df.copy()
        df[f"{date_column}_year"] = df[date_column].dt.year
        df[f"{date_column}_month"] = df[date_column].dt.month
        df[f"{date_column}_day"] = df[date_column].dt.day
        df[f"{date_column}_dayofweek"] = df[date_column].dt.dayofweek
        df[f"{date_column}_quarter"] = df[date_column].dt.quarter

        return df
    except Exception as e:
        logger.error(f"Error adding date components: {str(e)}")
        return df


def get_missing_dates(
    df: pd.DataFrame, date_column: str = "date", freq: str = "D"
) -> List[Any]:
    """Get list of missing dates in a time series."""
    if df.empty or date_column not in df.columns:
        return []

    try:
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = safe_to_datetime(df[date_column])

        # Get date range
        date_range = get_date_range(df, date_column)
        if not date_range:
            return []

        # Create complete date range
        all_dates = pd.date_range(start=date_range[0], end=date_range[1], freq=freq)

        # Get existing dates
        existing_dates = set(df[date_column].dropna())

        # Find missing dates
        missing_dates = [date for date in all_dates if date not in existing_dates]

        return missing_dates
    except Exception as e:
        logger.error(f"Error getting missing dates: {str(e)}")
        return []
