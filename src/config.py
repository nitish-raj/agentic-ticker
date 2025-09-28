"""
Centralized configuration management for Agentic Ticker.

This module provides a unified configuration system that works across all
utility modules while maintaining backward compatibility with existing
environment variable usage.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AssetType(str, Enum):
    """Asset type enumeration."""
    STOCK = "stock"
    CRYPTO = "crypto"
    AMBIGUOUS = "ambiguous"


@dataclass
class GeminiConfig:
    """Gemini API configuration."""
    api_key: str = ""
    model: str = "gemini-2.5-flash-lite"
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: int = 120

    def __post_init__(self):
        """Load from environment variables if not provided."""
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "")
        if not self.model:
            self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        if not self.api_base:
            self.api_base = os.getenv(
                "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
            )


@dataclass
class CoinGeckoConfig:
    """CoinGecko API configuration."""
    demo_api_key: str = ""
    pro_api_key: str = ""
    environment: str = "demo"
    timeout: int = 30

    def __post_init__(self):
        """Load from environment variables if not provided."""
        if not self.demo_api_key:
            self.demo_api_key = os.getenv("COINGECKO_DEMO_API_KEY", "")
        if not self.pro_api_key:
            self.pro_api_key = os.getenv("COINGECKO_API_KEY", "")


@dataclass
class YahooFinanceConfig:
    """Yahoo Finance configuration."""
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class DDGConfig:
    """DuckDuckGo Search configuration."""
    max_results: int = 3
    region: str = "us-en"
    safesearch: str = "moderate"
    timeout: int = 30


@dataclass
class AnalysisConfig:
    """Analysis parameters configuration."""
    default_days: int = 30
    default_threshold: float = 2.0
    default_forecast_days: int = 5
    max_analysis_steps: int = 10
    min_data_points: int = 5
    volatility_window: int = 10
    ma5_window: int = 5
    ma10_window: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5

    def __post_init__(self):
        """Load from environment variables if not provided."""
        if log_level := os.getenv("LOG_LEVEL"):
            try:
                self.level = LogLevel(log_level.upper())
            except ValueError:
                pass


@dataclass
class FeatureFlags:
    """Feature flags configuration."""
    enable_web_search: bool = True
    enable_crypto_analysis: bool = True
    enable_stock_analysis: bool = True
    enable_forecasting: bool = True
    enable_technical_indicators: bool = True
    enable_animations: bool = True
    enable_caching: bool = True
    enable_retry_logic: bool = True
    enable_error_handling: bool = True
    enable_validation: bool = True

    def __post_init__(self):
        """Load from environment variables if not provided."""
        flag_mappings = {
            "ENABLE_WEB_SEARCH": "enable_web_search",
            "ENABLE_CRYPTO_ANALYSIS": "enable_crypto_analysis",
            "ENABLE_STOCK_ANALYSIS": "enable_stock_analysis",
            "ENABLE_FORECASTING": "enable_forecasting",
            "ENABLE_TECHNICAL_INDICATORS": "enable_technical_indicators",
            "ENABLE_ANIMATIONS": "enable_animations",
            "ENABLE_CACHING": "enable_caching",
            "ENABLE_RETRY_LOGIC": "enable_retry_logic",
            "ENABLE_ERROR_HANDLING": "enable_error_handling",
            "ENABLE_VALIDATION": "enable_validation",
        }

        for env_var, attr_name in flag_mappings.items():
            if env_value := os.getenv(env_var):
                setattr(
                    self, attr_name, env_value.lower() in ("true", "1", "yes", "on")
                )


@dataclass
class UIConfig:
    """User interface configuration."""
    page_title: str = "Agentic-Ticker (Gemini)"
    page_icon: str = "📈"
    layout: str = "wide"
    chart_height: int = 500
    animation_duration: int = 500
    transition_duration: int = 300


@dataclass
class AppConfig:
    """Main application configuration."""
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    coingecko: CoinGeckoConfig = field(default_factory=CoinGeckoConfig)
    yahoo_finance: YahooFinanceConfig = field(default_factory=YahooFinanceConfig)
    ddg: DDGConfig = field(default_factory=DDGConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    ui: UIConfig = field(default_factory=UIConfig)

    # Configuration file settings
    config_file_path: Optional[str] = None
    hot_reload_enabled: bool = False
    hot_reload_interval: int = 60  # seconds

    def __post_init__(self):
        """Initialize configuration and load from file if available."""
        if self.config_file_path:
            self.load_from_file(self.config_file_path)
        else:
            # Try to find config file in common locations
            config_paths = [
                "config.json",
                "config.yaml",
                ".agentic-ticker.json",
                ".agentic-ticker.yaml",
                os.path.expanduser("~/.agentic-ticker.json"),
                os.path.expanduser("~/.agentic-ticker.yaml"),
            ]

            for path in config_paths:
                if os.path.exists(path):
                    self.load_from_file(path)
                    break

    def load_from_file(self, file_path: str) -> None:
        """Load configuration from JSON or YAML file."""
        path = Path(file_path)
        if not path.exists():
            return

        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            self.update_from_dict(data)

        except ImportError:
            # YAML not available, try JSON
            if path.suffix.lower() in ['.yaml', '.yml']:
                return
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self.update_from_dict(data)
            except Exception as e:
                logging.warning(f"Failed to load config from {file_path}: {e}")
        except Exception as e:
            logging.warning(f"Failed to load config from {file_path}: {e}")

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if not isinstance(data, dict):
            return

        # Update nested configurations
        for section_name, section_config in [
            ("gemini", self.gemini),
            ("coingecko", self.coingecko),
            ("yahoo_finance", self.yahoo_finance),
            ("ddg", self.ddg),
            ("analysis", self.analysis),
            ("logging", self.logging),
            ("feature_flags", self.feature_flags),
            ("ui", self.ui),
        ]:
            if section_name in data and isinstance(data[section_name], dict):
                for key, value in data[section_name].items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

        # Update top-level configuration
        for key, value in data.items():
            if key not in [s[0] for s in [
                ("gemini",), ("coingecko",), ("yahoo_finance",), ("ddg",),
                ("analysis",), ("logging",), ("feature_flags",), ("ui",),
            ]] and hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to file."""
        path = Path(file_path)
        data = self.to_dict()

        try:
            with open(path, 'w') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    yaml.dump(data, f, default_flow_style=False)
                else:
                    json.dump(data, f, indent=2)
        except ImportError:
            # YAML not available, use JSON
            if path.suffix.lower() in ['.yaml', '.yml']:
                return
            try:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to save config to {file_path}: {e}")
        except Exception as e:
            logging.error(f"Failed to save config to {file_path}: {e}")

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate Gemini configuration (optional - functions can work without it)
        # if not self.gemini.api_key:
        #     errors.append("GEMINI_API_KEY is required")

        if self.gemini.temperature < 0 or self.gemini.temperature > 2:
            errors.append("Gemini temperature must be between 0 and 2")

        if self.gemini.timeout <= 0:
            errors.append("Gemini timeout must be positive")

        # Validate analysis configuration
        if self.analysis.default_days <= 0:
            errors.append("Default days must be positive")

        if self.analysis.default_threshold <= 0:
            errors.append("Default threshold must be positive")

        if self.analysis.default_forecast_days <= 0:
            errors.append("Default forecast days must be positive")

        # Validate logging configuration
        if self.logging.file_path:
            log_dir = os.path.dirname(self.logging.file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except OSError:
                    errors.append(f"Cannot create log directory: {log_dir}")

        return errors

    def get_env_vars(self) -> Dict[str, str]:
        """Get environment variables representation of configuration."""
        env_vars = {}

        # Gemini configuration
        if self.gemini.api_key:
            env_vars["GEMINI_API_KEY"] = self.gemini.api_key
        if self.gemini.model:
            env_vars["GEMINI_MODEL"] = self.gemini.model
        if self.gemini.api_base:
            env_vars["GEMINI_API_BASE"] = self.gemini.api_base

        # CoinGecko configuration
        if self.coingecko.demo_api_key:
            env_vars["COINGECKO_DEMO_API_KEY"] = self.coingecko.demo_api_key
        if self.coingecko.pro_api_key:
            env_vars["COINGECKO_API_KEY"] = self.coingecko.pro_api_key

        # Logging configuration
        if self.logging.level:
            level_value = self.logging.level.value if hasattr(self.logging.level, 'value') else self.logging.level
            env_vars["LOG_LEVEL"] = level_value

        # Feature flags
        flag_mappings = {
            "enable_web_search": "ENABLE_WEB_SEARCH",
            "enable_crypto_analysis": "ENABLE_CRYPTO_ANALYSIS",
            "enable_stock_analysis": "ENABLE_STOCK_ANALYSIS",
            "enable_forecasting": "ENABLE_FORECASTING",
            "enable_technical_indicators": "ENABLE_TECHNICAL_INDICATORS",
            "enable_animations": "ENABLE_ANIMATIONS",
            "enable_caching": "ENABLE_CACHING",
            "enable_retry_logic": "ENABLE_RETRY_LOGIC",
            "enable_error_handling": "ENABLE_ERROR_HANDLING",
            "enable_validation": "ENABLE_VALIDATION",
        }

        for attr_name, env_var in flag_mappings.items():
            value = getattr(self.feature_flags, attr_name)
            env_vars[env_var] = str(value).lower()

        return env_vars


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(config_file_path: Optional[str] = None) -> AppConfig:
    """Load configuration from file and set as global instance."""
    config = AppConfig(config_file_path=config_file_path)

    # Validate configuration
    errors = config.validate()
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        if config.feature_flags.enable_error_handling:
            logging.error(error_msg)
        else:
            raise ValueError(error_msg)

    set_config(config)
    return config


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Set up logging based on configuration."""
    if config is None:
        config = get_config().logging

    # Configure root logger - handle both enum and string values
    level_value = config.level.value if hasattr(config.level, 'value') else config.level
    logging.basicConfig(
        level=getattr(logging, level_value.upper()),
        format=config.format,
        force=True
    )

    # Add file handler if file path is specified
    if config.file_path:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(logging.Formatter(config.format))
        
        # Handle both enum and string values for file handler level
        level_value = config.level.value if hasattr(config.level, 'value') else config.level
        file_handler.setLevel(getattr(logging, level_value.upper()))

        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with fallback to configuration."""
    # First try environment variable
    value = os.getenv(key)
    if value is not None:
        return value

    # Fallback to configuration
    config = get_config()
    env_vars = config.get_env_vars()
    return env_vars.get(key, default)


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature flag is enabled."""
    config = get_config()
    return getattr(config.feature_flags, feature_name, False)


def reload_config() -> None:
    """Reload configuration from file (hot reload)."""
    config = get_config()
    if config.config_file_path and config.hot_reload_enabled:
        load_config(config.config_file_path)
        setup_logging()


# Initialize configuration on module import
try:
    load_config()
    setup_logging()
except Exception as e:
    # Use basic logging if configuration fails
    logging.basicConfig(level=logging.INFO)
    logging.warning(f"Failed to initialize configuration: {e}")
