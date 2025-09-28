import functools
import time
import logging
from typing import Any, Callable, Dict, Optional, Type
import pandas as pd
import plotly.graph_objects as go


logger = logging.getLogger(__name__)


def handle_errors(
    default_return: Any = None,
    log_errors: bool = True,
    reraise_exceptions: Optional[Type[BaseException]] = None
):
    """Decorator for handling errors in functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                if reraise_exceptions and isinstance(e, reraise_exceptions):
                    raise
                # Return appropriate default based on expected return type
                if default_return is not None:
                    return default_return
                # Try to infer appropriate default return
                if func.__name__.startswith('create_') and 'chart' in func.__name__:
                    return go.Figure()
                elif (func.__name__.startswith('get_') or
                      func.__name__.startswith('find_')):
                    return None
                elif (func.__name__.startswith('process_') or
                      func.__name__.startswith('transform_')):
                    return args[0] if args else None
                return None
        return wrapper
    return decorator


def log_execution(include_args: bool = False, include_result: bool = False):
    """Decorator for logging function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            args_str = f"args={args}" if include_args else ""
            kwargs_str = f"kwargs={kwargs}" if include_args else ""
            logger.info(f"Executing {func.__name__}({args_str}, {kwargs_str})")
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                result_str = f" -> {result}" if include_result else ""
                logger.info(f"Completed {func.__name__} in "
                            f"{execution_time:.2f}s{result_str}")
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.error(f"Failed {func.__name__} after "
                             f"{execution_time:.2f}s: {str(e)}")
                raise
        return wrapper
    return decorator


def time_execution(log_threshold: Optional[float] = None):
    """Decorator for timing function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            if log_threshold is None or execution_time > log_threshold:
                logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        return wrapper
    return decorator


def validate_inputs(**validators):
    """Decorator for validating function inputs."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature for parameter mapping
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if validator == 'dataframe':
                        if not isinstance(value, pd.DataFrame):
                            raise TypeError(f"Parameter '{param_name}' must be a "
                                            f"pandas DataFrame")
                    elif validator == 'list_of_dicts':
                        if (not isinstance(value, list) or
                                not all(isinstance(item, dict) for item in value)):
                            raise TypeError(f"Parameter '{param_name}' must be a "
                                            f"list of dictionaries")
                    elif validator == 'non_empty_list':
                        if not isinstance(value, list) or len(value) == 0:
                            raise ValueError(f"Parameter '{param_name}' must be a "
                                             f"non-empty list")
                    elif validator == 'positive_number':
                        if not isinstance(value, (int, float)) or value <= 0:
                            raise ValueError(f"Parameter '{param_name}' must be a "
                                             f"positive number")
                    elif validator == 'non_empty_string':
                        if not isinstance(value, str) or not value.strip():
                            raise ValueError(f"Parameter '{param_name}' must be a "
                                             f"non-empty string")
                    elif callable(validator):
                        # Custom validation function
                        if not validator(value):
                            raise ValueError(f"Parameter '{param_name}' "
                                             f"failed validation")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_result(max_size: int = 128):
    """Simple caching decorator for functions with hashable arguments."""
    def decorator(func: Callable) -> Callable:

        cache: Dict[tuple, Any] = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Convert args to be hashable (lists to tuples, dicts to frozensets)
            def make_hashable(obj):
                if isinstance(obj, list):
                    return tuple(make_hashable(item) for item in obj)
                elif isinstance(obj, dict):
                    return frozenset((k, make_hashable(v)) for k, v in obj.items())
                elif isinstance(obj, set):
                    return frozenset(make_hashable(item) for item in obj)
                else:
                    return obj
            try:
                # Create hashable cache key
                hashable_args = tuple(make_hashable(arg) for arg in args)
                hashable_kwargs = frozenset((k, make_hashable(v))
                                            for k, v in kwargs.items())
                cache_key = (hashable_args, hashable_kwargs)
            except (TypeError, ValueError):
                # If conversion fails, don't cache
                return func(*args, **kwargs)

            if cache_key in cache:
                return cache[cache_key]
            result = func(*args, **kwargs)

            # Simple cache eviction (remove oldest if over max_size)
            if len(cache) >= max_size:
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            cache[cache_key] = result
            return result
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0,
                     exceptions: tuple = (Exception,)):
    """Decorator for retrying functions on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for "
                                       f"{func.__name__}: {str(e)}. "
                                       f"Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for "
                                     f"{func.__name__}")
            if last_exception is not None:
                raise last_exception
            else:
                raise RuntimeError(f"All {max_attempts} attempts failed for "
                                   f"{func.__name__}")
        return wrapper
    return decorator
