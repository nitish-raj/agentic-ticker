"""
Backward Compatibility Layer for Agentic Ticker

This module provides wrapper functions and compatibility mechanisms to ensure
existing code continues to work while migrating to the new utility modules
architecture.

Features:
- Wrapper functions maintaining existing signatures
- Deprecation warnings for old usage patterns
- Graceful fallback mechanisms
- Version compatibility checks
- Configuration options for compatibility mode
"""

import warnings
import functools
import inspect
import os
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Version information
COMPATIBILITY_VERSION = "1.0.0"
MIN_SUPPORTED_VERSION = "0.9.0"


# Compatibility mode configuration
class CompatibilityConfig:
    """Configuration for backward compatibility behavior"""

    def __init__(self):
        self.enabled = True  # Enable compatibility layer by default
        self.show_deprecation_warnings = True
        self.strict_mode = False  # If True, raises exceptions instead of warnings
        self.fallback_to_legacy = True  # Enable fallback to legacy implementations
        self.migration_deadline = datetime(2025, 12, 31)  # Deadline for migration

    @classmethod
    def from_env(cls) -> 'CompatibilityConfig':
        """Create configuration from environment variables"""
        config = cls()
        config.enabled = os.getenv(
            "COMPATIBILITY_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
        config.show_deprecation_warnings = os.getenv(
            "COMPATIBILITY_WARNINGS", "true"
        ).lower() in ("true", "1", "yes")
        config.strict_mode = os.getenv(
            "COMPATIBILITY_STRICT", "false"
        ).lower() in ("true", "1", "yes")
        config.fallback_to_legacy = os.getenv(
            "COMPATIBILITY_FALLBACK", "true"
        ).lower() in ("true", "1", "yes")

        # Parse migration deadline if provided
        deadline_str = os.getenv("COMPATIBILITY_DEADLINE")
        if deadline_str:
            try:
                config.migration_deadline = datetime.fromisoformat(deadline_str)
            except ValueError:
                logger.warning(
                    f"Invalid COMPATIBILITY_DEADLINE format: {deadline_str}"
                )

        return config


# Global configuration instance
compatibility_config = CompatibilityConfig.from_env()


def check_version_compatibility(version: str) -> bool:
    """
    Check if the provided version is compatible with current compatibility layer.

    Args:
        version: Version string to check

    Returns:
        True if compatible, False otherwise
    """
    try:
        import packaging.version
        current_version = packaging.version.parse(COMPATIBILITY_VERSION)
        min_version = packaging.version.parse(MIN_SUPPORTED_VERSION)
        provided_version = packaging.version.parse(version)

        return provided_version >= min_version and provided_version <= current_version
    except ImportError:
        # Fallback to simple string comparison if packaging not available
        return version >= MIN_SUPPORTED_VERSION


def deprecation_warning(message: str, stacklevel: int = 2) -> None:
    """
    Issue a deprecation warning with proper configuration handling.

    Args:
        message: Warning message
        stacklevel: Stack level for warning location
    """
    if not compatibility_config.show_deprecation_warnings:
        return

    if compatibility_config.strict_mode:
        raise DeprecationWarning(message)

    warnings.warn(
        f"[DEPRECATION] {message} (Compatibility layer v{COMPATIBILITY_VERSION})",
        DeprecationWarning,
        stacklevel=stacklevel
    )


def compatibility_wrapper(
    func: Callable,
    new_func: Optional[Callable] = None,
    migration_guide: Optional[str] = None
) -> Callable:
    """
    Decorator to wrap existing functions with compatibility layer.

    Args:
        func: Original function to wrap
        new_func: New function to delegate to (if None, uses same function)
        migration_guide: Optional migration guide URL or message

    Returns:
        Wrapped function with compatibility features
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if compatibility layer is enabled
        if not compatibility_config.enabled:
            return func(*args, **kwargs)

        # Issue deprecation warning
        func_name = func.__name__
        module_name = func.__module__

        warning_msg = f"{module_name}.{func_name} is deprecated"
        if new_func:
            new_module = new_func.__module__
            new_name = new_func.__name__
            warning_msg += f", use {new_module}.{new_name} instead"

        if migration_guide:
            warning_msg += f". See: {migration_guide}"

        if compatibility_config.migration_deadline:
            days_until_deadline = (
                compatibility_config.migration_deadline - datetime.now()
            ).days
            if days_until_deadline > 0:
                warning_msg += (
                    f" (Migration deadline: {days_until_deadline} days remaining)"
                )
            else:
                warning_msg += " (Migration deadline passed)"

        deprecation_warning(warning_msg, stacklevel=3)

        # Try to use new implementation if available
        if new_func and compatibility_config.fallback_to_legacy:
            try:
                return new_func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"New implementation failed for {func_name}, "
                    f"falling back to legacy: {e}"
                )
                return func(*args, **kwargs)

        # Use original function
        return func(*args, **kwargs)

    return wrapper


def create_function_signature_wrapper(
    original_func: Callable,
    new_func: Callable
) -> Callable:
    """
    Create a wrapper that maintains the original function signature but delegates
    to new function.

    Args:
        original_func: Original function with signature to maintain
        new_func: New function to delegate to

    Returns:
        Wrapper function with original signature
    """
    original_sig = inspect.signature(original_func)

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        # Bind arguments to original signature
        bound_args = original_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Convert to keyword arguments for new function
        kwargs_for_new = dict(bound_args.arguments)

        # Call new function
        return new_func(**kwargs_for_new)

    return wrapper


class CompatibilityRegistry:
    """Registry for managing backward compatibility mappings"""

    def __init__(self):
        self._legacy_to_new: Dict[str, str] = {}
        self._deprecated_functions: Dict[str, Callable] = {}
        self._migration_guides: Dict[str, str] = {}

    def register_compatibility(
        self,
        legacy_func_name: str,
        new_func_name: str,
        migration_guide: Optional[str] = None
    ) -> None:
        """
        Register a compatibility mapping between legacy and new functions.

        Args:
            legacy_func_name: Legacy function name (module.function format)
            new_func_name: New function name (module.function format)
            migration_guide: Optional migration guide
        """
        self._legacy_to_new[legacy_func_name] = new_func_name
        if migration_guide:
            self._migration_guides[legacy_func_name] = migration_guide

    def get_new_function_name(self, legacy_func_name: str) -> Optional[str]:
        """Get the new function name for a legacy function"""
        return self._legacy_to_new.get(legacy_func_name)

    def has_compatibility(self, legacy_func_name: str) -> bool:
        """Check if compatibility mapping exists for a function"""
        return legacy_func_name in self._legacy_to_new

    def get_migration_guide(self, legacy_func_name: str) -> Optional[str]:
        """Get migration guide for a function"""
        return self._migration_guides.get(legacy_func_name)

    def list_all_mappings(self) -> Dict[str, str]:
        """Get all compatibility mappings"""
        return self._legacy_to_new.copy()


# Global compatibility registry
compatibility_registry = CompatibilityRegistry()


def validate_compatibility_config() -> List[str]:
    """
    Validate the current compatibility configuration and return any issues.

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    # Check if migration deadline has passed
    if datetime.now() > compatibility_config.migration_deadline:
        issues.append(
            f"Migration deadline has passed: {compatibility_config.migration_deadline}"
        )

    # Check if compatibility is disabled but legacy functions are still being used
    if not compatibility_config.enabled:
        # This would require runtime checking, so just note the potential issue
        issues.append(
            "Compatibility layer is disabled - ensure all code has been migrated"
        )

    return issues


def get_compatibility_status() -> Dict[str, Any]:
    """
    Get current compatibility status and configuration.

    Returns:
        Dictionary with compatibility status information
    """
    return {
        "version": COMPATIBILITY_VERSION,
        "min_supported_version": MIN_SUPPORTED_VERSION,
        "enabled": compatibility_config.enabled,
        "show_deprecation_warnings": compatibility_config.show_deprecation_warnings,
        "strict_mode": compatibility_config.strict_mode,
        "fallback_to_legacy": compatibility_config.fallback_to_legacy,
        "migration_deadline": compatibility_config.migration_deadline.isoformat(),
        "days_until_deadline": max(
            0, (compatibility_config.migration_deadline - datetime.now()).days
        ),
        "validation_issues": validate_compatibility_config(),
        "registered_mappings": len(compatibility_registry.list_all_mappings())
    }


def enable_compatibility_warnings() -> None:
    """Enable deprecation warnings"""
    compatibility_config.show_deprecation_warnings = True


def disable_compatibility_warnings() -> None:
    """Disable deprecation warnings"""
    compatibility_config.show_deprecation_warnings = False


def set_strict_mode(strict: bool = True) -> None:
    """
    Enable or disable strict mode.

    Args:
        strict: If True, raises exceptions instead of warnings
    """
    compatibility_config.strict_mode = strict


def set_migration_deadline(deadline: Union[str, datetime]) -> None:
    """
    Set the migration deadline.

    Args:
        deadline: Deadline as datetime object or ISO string
    """
    if isinstance(deadline, str):
        deadline = datetime.fromisoformat(deadline)

    compatibility_config.migration_deadline = deadline


# Initialize compatibility registry with common mappings
def _initialize_compatibility_mappings():
    """Initialize default compatibility mappings"""
    mappings = [
        ("services.validate_ticker", "utility_modules.validation.validate_ticker"),
        ("services.get_company_info", "utility_modules.company_info.get_company_info"),
        ("services.get_crypto_info", "utility_modules.crypto_info.get_crypto_info"),
        ("services.load_prices",
         "utility_modules.data_loading.load_prices"),
        ("services.load_crypto_prices",
         "utility_modules.data_loading.load_crypto_prices"),
        ("services.compute_indicators",
         "utility_modules.analysis.compute_indicators"),
        ("services.detect_events", "utility_modules.analysis.detect_events"),
        ("services.forecast_prices", "utility_modules.analysis.forecast_prices"),
        ("services.build_report", "utility_modules.reporting.build_report"),
        ("services.ddgs_search", "utility_modules.search.ddgs_search"),
    ]

    for legacy, new in mappings:
        compatibility_registry.register_compatibility(
            legacy,
            new,
            migration_guide="See MIGRATION_GUIDE.md for detailed migration instructions"
        )


# Initialize mappings on module import
_initialize_compatibility_mappings()
