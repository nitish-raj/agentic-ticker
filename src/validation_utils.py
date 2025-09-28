import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
import logging

logger = logging.getLogger(__name__)


def clean_numeric_data(
    series: pd.Series, 
    default_value: float = 0.0,
    remove_outliers: bool = False,
    outlier_threshold: float = 3.0
) -> Union[pd.Series, pd.DataFrame]:
    """Clean numeric data by handling None/NaN values and optionally removing outliers."""
    if series.empty:
        return series
    
    # Handle None/NaN values
    cleaned_series = series.apply(lambda x: default_value if pd.isna(x) or x is None else x)
    
    # Remove outliers if requested
    if remove_outliers and len(cleaned_series) > 0:
        mean_val = cleaned_series.mean()
        std_val = cleaned_series.std()
        
        if std_val > 0:  # Avoid division by zero
            z_scores = np.abs((cleaned_series - mean_val) / std_val)
            cleaned_series = cleaned_series[z_scores <= outlier_threshold]
    
    return cleaned_series  # type: ignore


def validate_dataframe(
    df: pd.DataFrame, 
    required_columns: Optional[List[str]] = None,
    min_rows: int = 0,
    max_rows: Optional[int] = None,
    column_types: Optional[Dict[str, str]] = None
) -> bool:
    """Validate DataFrame structure and content."""
    if df.empty and min_rows > 0:
        logger.error("DataFrame is empty but minimum rows required")
        return False
    
    # Check row count
    if len(df) < min_rows:
        logger.error(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
        return False
    
    if max_rows is not None and len(df) > max_rows:
        logger.error(f"DataFrame has {len(df)} rows, maximum {max_rows} allowed")
        return False
    
    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
    
    # Check column types
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[col]):
                    logger.error(f"Column '{col}' should be numeric but is {df[col].dtype}")
                    return False
                elif expected_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    logger.error(f"Column '{col}' should be datetime but is {df[col].dtype}")
                    return False
                elif expected_type == 'string' and not pd.api.types.is_string_dtype(df[col]):
                    logger.error(f"Column '{col}' should be string but is {df[col].dtype}")
                    return False
    
    return True


def sanitize_forecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate forecast-specific data."""
    if df.empty:
        return df
    
    sanitized_df = df.copy()
    
    # Clean confidence values
    if 'confidence' in sanitized_df.columns:
        sanitized_df['confidence'] = sanitized_df['confidence'].apply(
            lambda x: 0.5 if pd.isna(x) or x is None else max(0.0, min(1.0, float(x)))
        )
    
    # Clean forecast prices
    if 'forecast_price' in sanitized_df.columns:
        sanitized_df['forecast_price'] = sanitized_df['forecast_price'].apply(
            lambda x: 0.0 if pd.isna(x) or x is None else float(x)
        )
    
    # Clean trend values
    if 'trend' in sanitized_df.columns:
        valid_trends = ['UP', 'DOWN', 'NEUTRAL']
        sanitized_df['trend'] = sanitized_df['trend'].apply(
            lambda x: 'NEUTRAL' if pd.isna(x) or x is None or x not in valid_trends else str(x).upper()
        )
    
    # Drop rows with invalid data
    sanitized_df = sanitized_df.dropna()
    
    return sanitized_df


def validate_list_of_dicts(
    data: List[Dict[str, Any]], 
    required_keys: Optional[List[str]] = None,
    min_length: int = 0,
    max_length: Optional[int] = None
) -> bool:
    """Validate a list of dictionaries."""
    if not isinstance(data, list):
        logger.error("Data is not a list")
        return False
    
    if len(data) < min_length:
        logger.error(f"List has {len(data)} items, minimum {min_length} required")
        return False
    
    if max_length is not None and len(data) > max_length:
        logger.error(f"List has {len(data)} items, maximum {max_length} allowed")
        return False
    
    # Check that all items are dictionaries
    if not all(isinstance(item, dict) for item in data):
        logger.error("Not all items in list are dictionaries")
        return False
    
    # Check required keys
    if required_keys:
        for i, item in enumerate(data):
            missing_keys = set(required_keys) - set(item.keys())
            if missing_keys:
                logger.error(f"Item {i} missing required keys: {missing_keys}")
                return False
    
    return True


def clean_string_data(
    series: pd.Series, 
    default_value: str = "",
    strip_whitespace: bool = True,
    convert_to_upper: bool = False,
    convert_to_lower: bool = False
) -> Union[pd.Series, pd.DataFrame]:
    """Clean string data by handling None/NaN values and applying transformations."""
    if series.empty:
        return series
    
    def clean_string(value: Any) -> str:
        if pd.isna(value) or value is None:
            return default_value
        
        str_value = str(value)
        
        if strip_whitespace:
            str_value = str_value.strip()
        
        if convert_to_upper:
            str_value = str_value.upper()
        elif convert_to_lower:
            str_value = str_value.lower()
        
        return str_value
    
    return series.apply(clean_string)


def validate_numeric_range(
    series: pd.Series, 
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_nan: bool = False
) -> Union[pd.Series, pd.DataFrame]:
    """Validate that numeric values are within specified range."""
    if series.empty:
        return series
    
    validated_series = series.copy()
    
    if not allow_nan:
        validated_series = validated_series.fillna(0.0)
    
    if min_value is not None:
        validated_series = validated_series.apply(lambda x: max(min_value, x) if pd.notna(x) else x)
    
    if max_value is not None:
        validated_series = validated_series.apply(lambda x: min(max_value, x) if pd.notna(x) else x)
    
    return validated_series


def remove_duplicates(
    df: pd.DataFrame, 
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """Remove duplicate rows from DataFrame."""
    if df.empty:
        return df
    
    try:
        return df.drop_duplicates(subset=subset, keep=keep)  # type: ignore
    except Exception as e:
        logger.error(f"Error removing duplicates: {str(e)}")
        return df


def validate_data_types(
    data: Any, 
    expected_type: Union[type, tuple],
    allow_none: bool = False
) -> bool:
    """Validate that data matches expected type."""
    if allow_none and data is None:
        return True
    
    if not isinstance(data, expected_type):
        logger.error(f"Data type {type(data)} does not match expected type {expected_type}")
        return False
    
    return True


def apply_custom_validation(
    df: pd.DataFrame, 
    validation_func: Callable[[pd.DataFrame], bool],
    error_message: str = "Custom validation failed"
) -> bool:
    """Apply custom validation function to DataFrame."""
    try:
        if not validation_func(df):
            logger.error(error_message)
            return False
        return True
    except Exception as e:
        logger.error(f"Error in custom validation: {str(e)}")
        return False