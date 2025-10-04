"""
Centralized configuration management for Agentic Ticker.

This module provides a unified configuration system that works across all
utility modules while maintaining backward compatibility with existing
environment variable usage.
"""

import os
import logging
import ipaddress
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

try:
    from dotenv import load_dotenv
    
    # Try to load .env file from current directory and parent directories
    load_dotenv()
except ImportError:
    # python-dotenv not available, continue without it
    pass


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


@dataclass
class CoinGeckoConfig:
    """CoinGecko API configuration."""

    demo_api_key: str = ""
    pro_api_key: str = ""
    environment: str = "demo"
    timeout: int = 30


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
class CORSConfig:
    """CORS configuration for FastAPI backend."""

    allowed_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:8501",  # Default Streamlit local port
            "http://localhost:3000",  # Common development port
            "http://127.0.0.1:8501",  # Alternative localhost
            "http://127.0.0.1:3000",  # Alternative localhost
        ]
    )
    allowed_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    allowed_headers: List[str] = field(
        default_factory=lambda: ["Content-Type", "Authorization", "X-Requested-With"]
    )
    allow_credentials: bool = True
    max_age: int = 600  # 10 minutes
    strict_origins: bool = True  # Enforce strict origin validation


@dataclass
class SecurityConfig:
    """Security configuration for network and application security."""

    # HTTPS/TLS settings
    https_enabled: bool = True
    tls_version: str = "TLSv1.3"
    verify_certificates: bool = True
    certificate_bundle_path: Optional[str] = None

    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10

    # Request limits
    max_request_size_mb: int = 10
    max_response_size_mb: int = 50

    # IP filtering
    whitelist_ips: List[str] = field(default_factory=list)
    blacklist_ips: List[str] = field(default_factory=list)

    # Security headers
    security_headers_enabled: bool = True
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True

    # Content Security Policy
    csp_enabled: bool = True
    csp_policy: str = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self' https://api.coingecko.com https://generativelanguage.googleapis.com; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )

    # Input validation
    input_validation_enabled: bool = True
    max_url_length: int = 2048
    max_header_length: int = 8192

    # Network security
    connection_timeout: int = 30
    read_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

    # DNS security
    dns_over_https: bool = False
    dnssec_enabled: bool = True
    trusted_dns_servers: List[str] = field(
        default_factory=lambda: [
            "8.8.8.8",
            "8.8.4.4",  # Google
            "1.1.1.1",
            "1.0.0.1",  # Cloudflare
        ]
    )

    # Monitoring and logging
    security_logging_enabled: bool = True
    log_failed_requests: bool = True
    log_blocked_ips: bool = True
    log_rate_limits: bool = True


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
    cors: CORSConfig = field(default_factory=CORSConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # Configuration settings
    hot_reload_enabled: bool = False
    hot_reload_interval: int = 60  # seconds

    def __post_init__(self):
        """Initialize configuration from environment variables only."""
        # Load from environment variables
        self.load_from_env()

    

    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Gemini API Configuration
        if os.getenv("GEMINI_API_KEY"):
            self.gemini.api_key = os.getenv("GEMINI_API_KEY", "")
        if os.getenv("GEMINI_MODEL"):
            self.gemini.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        if os.getenv("GEMINI_API_BASE"):
            self.gemini.api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        if os.getenv("GEMINI_TEMPERATURE"):
            try:
                self.gemini.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
            except ValueError:
                pass
        if os.getenv("GEMINI_MAX_TOKENS"):
            try:
                self.gemini.max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
            except ValueError:
                pass
        if os.getenv("GEMINI_TIMEOUT"):
            try:
                self.gemini.timeout = int(os.getenv("GEMINI_TIMEOUT", "120"))
            except ValueError:
                pass

        # CoinGecko API Configuration
        if os.getenv("COINGECKO_DEMO_API_KEY"):
            self.coingecko.demo_api_key = os.getenv("COINGECKO_DEMO_API_KEY", "")
        if os.getenv("COINGECKO_API_KEY"):
            self.coingecko.pro_api_key = os.getenv("COINGECKO_API_KEY", "")
        if os.getenv("COINGECKO_ENVIRONMENT"):
            self.coingecko.environment = os.getenv("COINGECKO_ENVIRONMENT", "demo")
        if os.getenv("COINGECKO_TIMEOUT"):
            try:
                self.coingecko.timeout = int(os.getenv("COINGECKO_TIMEOUT", "30"))
            except ValueError:
                pass

        # Yahoo Finance Configuration
        if os.getenv("YAHOO_FINANCE_TIMEOUT"):
            try:
                self.yahoo_finance.timeout = int(os.getenv("YAHOO_FINANCE_TIMEOUT", "30"))
            except ValueError:
                pass
        if os.getenv("YAHOO_FINANCE_RETRY_ATTEMPTS"):
            try:
                self.yahoo_finance.retry_attempts = int(os.getenv("YAHOO_FINANCE_RETRY_ATTEMPTS", "3"))
            except ValueError:
                pass
        if os.getenv("YAHOO_FINANCE_RETRY_DELAY"):
            try:
                self.yahoo_finance.retry_delay = float(os.getenv("YAHOO_FINANCE_RETRY_DELAY", "1.0"))
            except ValueError:
                pass

        # DuckDuckGo Search Configuration
        if os.getenv("DDG_MAX_RESULTS"):
            try:
                self.ddg.max_results = int(os.getenv("DDG_MAX_RESULTS", "3"))
            except ValueError:
                pass
        if os.getenv("DDG_REGION"):
            self.ddg.region = os.getenv("DDG_REGION", "us-en")
        if os.getenv("DDG_SAFESEARCH"):
            self.ddg.safesearch = os.getenv("DDG_SAFESEARCH", "moderate")
        if os.getenv("DDG_TIMEOUT"):
            try:
                self.ddg.timeout = int(os.getenv("DDG_TIMEOUT", "30"))
            except ValueError:
                pass

        # Analysis Parameters
        if os.getenv("ANALYSIS_DEFAULT_DAYS"):
            try:
                self.analysis.default_days = int(os.getenv("ANALYSIS_DEFAULT_DAYS", "30"))
            except ValueError:
                pass
        if os.getenv("ANALYSIS_DEFAULT_THRESHOLD"):
            try:
                self.analysis.default_threshold = float(os.getenv("ANALYSIS_DEFAULT_THRESHOLD", "2.0"))
            except ValueError:
                pass
        if os.getenv("ANALYSIS_DEFAULT_FORECAST_DAYS"):
            try:
                self.analysis.default_forecast_days = int(os.getenv("ANALYSIS_DEFAULT_FORECAST_DAYS", "5"))
            except ValueError:
                pass
        if os.getenv("ANALYSIS_MAX_ANALYSIS_STEPS"):
            try:
                self.analysis.max_analysis_steps = int(os.getenv("ANALYSIS_MAX_ANALYSIS_STEPS", "10"))
            except ValueError:
                pass
        if os.getenv("ANALYSIS_MIN_DATA_POINTS"):
            try:
                self.analysis.min_data_points = int(os.getenv("ANALYSIS_MIN_DATA_POINTS", "5"))
            except ValueError:
                pass
        if os.getenv("ANALYSIS_VOLATILITY_WINDOW"):
            try:
                self.analysis.volatility_window = int(os.getenv("ANALYSIS_VOLATILITY_WINDOW", "10"))
            except ValueError:
                pass
        if os.getenv("ANALYSIS_MA5_WINDOW"):
            try:
                self.analysis.ma5_window = int(os.getenv("ANALYSIS_MA5_WINDOW", "5"))
            except ValueError:
                pass
        if os.getenv("ANALYSIS_MA10_WINDOW"):
            try:
                self.analysis.ma10_window = int(os.getenv("ANALYSIS_MA10_WINDOW", "10"))
            except ValueError:
                pass

        # Logging Configuration
        if os.getenv("LOGGING_LEVEL"):
            level_str = os.getenv("LOGGING_LEVEL", "INFO").upper()
            try:
                self.logging.level = LogLevel(level_str)
            except ValueError:
                # If invalid level, use default
                pass
        if os.getenv("LOGGING_FORMAT"):
            self.logging.format = os.getenv("LOGGING_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        if os.getenv("LOGGING_FILE_PATH"):
            self.logging.file_path = os.getenv("LOGGING_FILE_PATH") or None
        if os.getenv("LOGGING_MAX_FILE_SIZE"):
            try:
                self.logging.max_file_size = int(os.getenv("LOGGING_MAX_FILE_SIZE", "10485760"))
            except ValueError:
                pass
        if os.getenv("LOGGING_BACKUP_COUNT"):
            try:
                self.logging.backup_count = int(os.getenv("LOGGING_BACKUP_COUNT", "5"))
            except ValueError:
                pass

        # Feature Flags
        if os.getenv("ENABLE_WEB_SEARCH"):
            self.feature_flags.enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
        if os.getenv("ENABLE_CRYPTO_ANALYSIS"):
            self.feature_flags.enable_crypto_analysis = os.getenv("ENABLE_CRYPTO_ANALYSIS", "true").lower() == "true"
        if os.getenv("ENABLE_STOCK_ANALYSIS"):
            self.feature_flags.enable_stock_analysis = os.getenv("ENABLE_STOCK_ANALYSIS", "true").lower() == "true"
        if os.getenv("ENABLE_FORECASTING"):
            self.feature_flags.enable_forecasting = os.getenv("ENABLE_FORECASTING", "true").lower() == "true"
        if os.getenv("ENABLE_TECHNICAL_INDICATORS"):
            self.feature_flags.enable_technical_indicators = os.getenv("ENABLE_TECHNICAL_INDICATORS", "true").lower() == "true"
        if os.getenv("ENABLE_ANIMATIONS"):
            self.feature_flags.enable_animations = os.getenv("ENABLE_ANIMATIONS", "true").lower() == "true"
        if os.getenv("ENABLE_CACHING"):
            self.feature_flags.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        if os.getenv("ENABLE_RETRY_LOGIC"):
            self.feature_flags.enable_retry_logic = os.getenv("ENABLE_RETRY_LOGIC", "true").lower() == "true"
        if os.getenv("ENABLE_ERROR_HANDLING"):
            self.feature_flags.enable_error_handling = os.getenv("ENABLE_ERROR_HANDLING", "true").lower() == "true"
        if os.getenv("ENABLE_VALIDATION"):
            self.feature_flags.enable_validation = os.getenv("ENABLE_VALIDATION", "true").lower() == "true"

        # User Interface Configuration
        if os.getenv("UI_PAGE_TITLE"):
            self.ui.page_title = os.getenv("UI_PAGE_TITLE", "Agentic-Ticker (Gemini)")
        if os.getenv("UI_PAGE_ICON"):
            self.ui.page_icon = os.getenv("UI_PAGE_ICON", "📈")
        if os.getenv("UI_LAYOUT"):
            self.ui.layout = os.getenv("UI_LAYOUT", "wide")
        if os.getenv("UI_CHART_HEIGHT"):
            try:
                self.ui.chart_height = int(os.getenv("UI_CHART_HEIGHT", "500"))
            except ValueError:
                pass
        if os.getenv("UI_ANIMATION_DURATION"):
            try:
                self.ui.animation_duration = int(os.getenv("UI_ANIMATION_DURATION", "500"))
            except ValueError:
                pass
        if os.getenv("UI_TRANSITION_DURATION"):
            try:
                self.ui.transition_duration = int(os.getenv("UI_TRANSITION_DURATION", "300"))
            except ValueError:
                pass

        # CORS Configuration
        if os.getenv("CORS_ALLOWED_ORIGINS"):
            origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
            self.cors.allowed_origins = [origin.strip() for origin in origins]
        if os.getenv("CORS_ALLOWED_METHODS"):
            methods = os.getenv("CORS_ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
            self.cors.allowed_methods = [method.strip() for method in methods]
        if os.getenv("CORS_ALLOWED_HEADERS"):
            headers = os.getenv("CORS_ALLOWED_HEADERS", "Content-Type,Authorization,X-Requested-With").split(",")
            self.cors.allowed_headers = [header.strip() for header in headers]
        if os.getenv("CORS_ALLOW_CREDENTIALS"):
            self.cors.allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
        if os.getenv("CORS_MAX_AGE"):
            try:
                self.cors.max_age = int(os.getenv("CORS_MAX_AGE", "600"))
            except ValueError:
                pass

        # Configuration Settings
        if os.getenv("CONFIG_FILE_PATH"):
            self.config_file_path = os.getenv("CONFIG_FILE_PATH") or None
        if os.getenv("HOT_RELOAD_ENABLED"):
            self.hot_reload_enabled = os.getenv("HOT_RELOAD_ENABLED", "false").lower() == "true"
        if os.getenv("HOT_RELOAD_INTERVAL"):
            try:
                self.hot_reload_interval = int(os.getenv("HOT_RELOAD_INTERVAL", "60"))
            except ValueError:
                pass

        # Security Settings (only update existing attributes)
        if os.getenv("SECURITY_HTTPS_ENABLED"):
            self.security.https_enabled = os.getenv("SECURITY_HTTPS_ENABLED", "true").lower() == "true"
        if os.getenv("SECURITY_TLS_VERSION"):
            self.security.tls_version = os.getenv("SECURITY_TLS_VERSION", "TLSv1.3")
        if os.getenv("SECURITY_VERIFY_CERTIFICATES"):
            self.security.verify_certificates = os.getenv("SECURITY_VERIFY_CERTIFICATES", "true").lower() == "true"
        if os.getenv("SECURITY_RATE_LIMIT_ENABLED"):
            self.security.rate_limit_enabled = os.getenv("SECURITY_RATE_LIMIT_ENABLED", "true").lower() == "true"
        if os.getenv("SECURITY_REQUESTS_PER_MINUTE"):
            try:
                self.security.requests_per_minute = int(os.getenv("SECURITY_REQUESTS_PER_MINUTE", "60"))
            except ValueError:
                pass
        if os.getenv("SECURITY_REQUESTS_PER_HOUR"):
            try:
                self.security.requests_per_hour = int(os.getenv("SECURITY_REQUESTS_PER_HOUR", "1000"))
            except ValueError:
                pass
        if os.getenv("SECURITY_CONNECTION_TIMEOUT"):
            try:
                self.security.connection_timeout = int(os.getenv("SECURITY_CONNECTION_TIMEOUT", "30"))
            except ValueError:
                pass
        if os.getenv("SECURITY_READ_TIMEOUT"):
            try:
                self.security.read_timeout = int(os.getenv("SECURITY_READ_TIMEOUT", "60"))
            except ValueError:
                pass

    

    

    def validate(self) -> tuple[List[str], List[str]]:
        """Validate configuration and return (errors, warnings)."""
        errors = []
        warnings = []

        # Check if validation is enabled
        if not self.feature_flags.enable_validation:
            return errors, warnings

        # Determine if we're in demo/live mode based on available API keys
        has_gemini_key = bool(self.gemini.api_key)
        has_coingecko_key = bool(
            self.coingecko.demo_api_key or self.coingecko.pro_api_key
        )
        live_mode = has_gemini_key or has_coingecko_key

        # Validate Gemini configuration
        if self.gemini.temperature < 0 or self.gemini.temperature > 2:
            errors.append("Gemini temperature must be between 0 and 2")

        if self.gemini.timeout <= 0:
            errors.append("Gemini timeout must be positive")

        if self.gemini.max_tokens <= 0:
            errors.append("Gemini max_tokens must be positive")

        # Validate Gemini API base URL format
        if self.gemini.api_base and not self._is_valid_url(self.gemini.api_base):
            errors.append("Gemini API base URL is invalid")

        # Gemini API key validation (conditional based on mode)
        if not has_gemini_key:
            if live_mode:
                errors.append(
                    "GEMINI_API_KEY is required for live mode. "
                    "Set it via environment variable or config file."
                )
            else:
                warnings.append(
                    "GEMINI_API_KEY not set - running in demo mode. "
                    "Some features will be limited."
                )

        # Validate CoinGecko configuration
        if self.coingecko.timeout <= 0:
            errors.append("CoinGecko timeout must be positive")

        if not has_coingecko_key and self.feature_flags.enable_crypto_analysis:
            if live_mode:
                warnings.append(
                    "CoinGecko API key not set - crypto analysis will use "
                    "demo mode with rate limits."
                )
            else:
                warnings.append(
                    "CoinGecko API key not set - crypto features will be "
                    "limited in demo mode."
                )

        # Validate Yahoo Finance configuration
        if self.yahoo_finance.timeout <= 0:
            errors.append("Yahoo Finance timeout must be positive")

        if self.yahoo_finance.retry_attempts < 0:
            errors.append("Yahoo Finance retry_attempts must be non-negative")

        if self.yahoo_finance.retry_delay < 0:
            errors.append("Yahoo Finance retry_delay must be non-negative")

        # Validate DDG configuration
        if self.ddg.max_results <= 0:
            errors.append("DuckDuckGo max_results must be positive")

        if self.ddg.timeout <= 0:
            errors.append("DuckDuckGo timeout must be positive")

        # Validate analysis configuration
        if self.analysis.default_days <= 0:
            errors.append("Default days must be positive")

        if self.analysis.default_threshold <= 0:
            errors.append("Default threshold must be positive")

        if self.analysis.default_forecast_days <= 0:
            errors.append("Default forecast days must be positive")

        if self.analysis.max_analysis_steps <= 0:
            errors.append("Max analysis steps must be positive")

        if self.analysis.min_data_points <= 0:
            errors.append("Min data points must be positive")

        if self.analysis.volatility_window <= 0:
            errors.append("Volatility window must be positive")

        if self.analysis.ma5_window <= 0:
            errors.append("MA5 window must be positive")

        if self.analysis.ma10_window <= 0:
            errors.append("MA10 window must be positive")

        # Validate logging configuration
        if self.logging.file_path:
            log_dir = os.path.dirname(self.logging.file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except OSError:
                    errors.append(f"Cannot create log directory: {log_dir}")

            if self.logging.max_file_size <= 0:
                errors.append("Max file size must be positive")

            if self.logging.backup_count < 0:
                errors.append("Backup count must be non-negative")

        # Validate UI configuration
        if self.ui.chart_height <= 0:
            errors.append("Chart height must be positive")

        if self.ui.animation_duration < 0:
            errors.append("Animation duration must be non-negative")

        if self.ui.transition_duration < 0:
            errors.append("Transition duration must be non-negative")

        # Validate CORS configuration
        if not self.cors.allowed_origins:
            errors.append("CORS allowed_origins cannot be empty")

        if not self.cors.allowed_methods:
            errors.append("CORS allowed_methods cannot be empty")

        if self.cors.max_age < 0:
            errors.append("CORS max_age must be non-negative")

        # Validate security configuration
        if self.security.requests_per_minute <= 0:
            errors.append("Security requests_per_minute must be positive")

        if self.security.requests_per_hour <= 0:
            errors.append("Security requests_per_hour must be positive")

        if self.security.burst_size <= 0:
            errors.append("Security burst_size must be positive")

        if self.security.max_request_size_mb <= 0:
            errors.append("Security max_request_size_mb must be positive")

        if self.security.max_response_size_mb <= 0:
            errors.append("Security max_response_size_mb must be positive")

        if self.security.connection_timeout <= 0:
            errors.append("Security connection_timeout must be positive")

        if self.security.read_timeout <= 0:
            errors.append("Security read_timeout must be positive")

        if self.security.max_retries < 0:
            errors.append("Security max_retries must be non-negative")

        if self.security.retry_delay < 0:
            errors.append("Security retry_delay must be non-negative")

        if self.security.max_url_length <= 0:
            errors.append("Security max_url_length must be positive")

        if self.security.max_header_length <= 0:
            errors.append("Security max_header_length must be positive")

        # Validate TLS version
        valid_tls_versions = ["TLSv1.2", "TLSv1.3"]
        if self.security.tls_version not in valid_tls_versions:
            errors.append(f"Security tls_version must be one of: {valid_tls_versions}")

        # Validate IP addresses
        for ip in self.security.whitelist_ips + self.security.blacklist_ips:
            try:
                ipaddress.ip_network(ip, strict=False)
            except ValueError:
                try:
                    ipaddress.ip_address(ip)
                except ValueError:
                    errors.append(f"Invalid IP address or network: {ip}")

        # Validate configuration file settings
        if self.hot_reload_interval <= 0:
            errors.append("Hot reload interval must be positive")

        # Feature-specific validation
        if self.feature_flags.enable_forecasting and not has_gemini_key:
            warnings.append(
                "Forecasting is enabled but Gemini API key is not set - "
                "forecasting will use mock data."
            )

        if self.feature_flags.enable_web_search and not self.ddg.max_results:
            warnings.append(
                "Web search is enabled but max_results is 0 - "
                "search will return no results."
            )

        return errors, warnings

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid."""
        import re

        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        return url_pattern.match(url) is not None

    def validate_runtime_change(
        self, section: str, key: str, value: Any
    ) -> tuple[bool, str]:
        """Validate a runtime configuration change."""
        try:
            # Validate based on section and key
            if section == "gemini":
                if key == "temperature":
                    if not (0 <= float(value) <= 2):
                        return False, "Gemini temperature must be between 0 and 2"
                elif key == "timeout":
                    if int(value) <= 0:
                        return False, "Gemini timeout must be positive"
                elif key == "max_tokens":
                    if int(value) <= 0:
                        return False, "Gemini max_tokens must be positive"
                elif key == "api_base":
                    if not self._is_valid_url(str(value)):
                        return False, "Gemini API base URL is invalid"

            elif section == "analysis":
                if key in [
                    "default_days",
                    "default_forecast_days",
                    "max_analysis_steps",
                    "min_data_points",
                    "volatility_window",
                    "ma5_window",
                    "ma10_window",
                ]:
                    if int(value) <= 0:
                        return False, f"{key} must be positive"
                elif key == "default_threshold":
                    if float(value) <= 0:
                        return False, "Default threshold must be positive"

            elif section == "ui":
                if key == "chart_height":
                    if int(value) <= 0:
                        return False, "Chart height must be positive"
                elif key in ["animation_duration", "transition_duration"]:
                    if int(value) < 0:
                        return False, f"{key} must be non-negative"

            elif section == "cors":
                if key == "max_age":
                    if int(value) < 0:
                        return False, "CORS max_age must be non-negative"
                elif key == "allowed_origins" and not value:
                    return False, "CORS allowed_origins cannot be empty"
                elif key == "allowed_methods" and not value:
                    return False, "CORS allowed_methods cannot be empty"

            elif section == "security":
                if key in [
                    "requests_per_minute",
                    "requests_per_hour",
                    "burst_size",
                    "max_request_size_mb",
                    "max_response_size_mb",
                    "connection_timeout",
                    "read_timeout",
                    "max_url_length",
                    "max_header_length",
                ]:
                    if int(value) <= 0:
                        return False, f"Security {key} must be positive"
                elif key in ["max_retries", "retry_delay"]:
                    if float(value) < 0:
                        return False, f"Security {key} must be non-negative"
                elif key == "tls_version":
                    valid_versions = ["TLSv1.2", "TLSv1.3"]
                    if value not in valid_versions:
                        return (
                            False,
                            f"Security tls_version must be one of: {valid_versions}",
                        )

            return True, "Valid"
        except (ValueError, TypeError) as e:
            return False, f"Invalid value type: {str(e)}"

    def get_mode_info(self) -> Dict[str, Any]:
        """Get information about current mode and available features."""
        has_gemini_key = bool(self.gemini.api_key)
        has_coingecko_key = bool(
            self.coingecko.demo_api_key or self.coingecko.pro_api_key
        )

        mode = "demo"
        if has_gemini_key and has_coingecko_key:
            mode = "full"
        elif has_gemini_key or has_coingecko_key:
            mode = "partial"

        available_features = []
        limited_features = []

        if has_gemini_key:
            available_features.extend(
                ["AI Analysis", "Forecasting", "Smart Validation"]
            )
        else:
            limited_features.append("AI Analysis (mock data)")
            limited_features.append("Forecasting (mock data)")

        if has_coingecko_key:
            available_features.append("Crypto Analysis (full)")
        else:
            limited_features.append("Crypto Analysis (demo/rate limited)")

        available_features.extend(["Stock Analysis", "Technical Indicators"])

        return {
            "mode": mode,
            "has_gemini_key": has_gemini_key,
            "has_coingecko_key": has_coingecko_key,
            "available_features": available_features,
            "limited_features": limited_features,
            "enable_validation": self.feature_flags.enable_validation,
        }

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
            level_value = (
                self.logging.level.value
                if hasattr(self.logging.level, "value")
                else self.logging.level
            )
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

        # CORS configuration
        if self.cors.allowed_origins:
            env_vars["CORS_ALLOWED_ORIGINS"] = ",".join(self.cors.allowed_origins)
        if self.cors.allowed_methods:
            env_vars["CORS_ALLOWED_METHODS"] = ",".join(self.cors.allowed_methods)
        if self.cors.allowed_headers:
            env_vars["CORS_ALLOWED_HEADERS"] = ",".join(self.cors.allowed_headers)
        env_vars["CORS_ALLOW_CREDENTIALS"] = str(self.cors.allow_credentials).lower()
        env_vars["CORS_MAX_AGE"] = str(self.cors.max_age)

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


def load_config() -> AppConfig:
    """Load configuration from environment variables and set as global instance."""
    config = AppConfig()

    # Validate configuration
    errors, warnings = config.validate()

    # Handle warnings first
    if warnings:
        warning_msg = "Configuration warnings:\n" + "\n".join(
            f"  - {warning}" for warning in warnings
        )
        logging.warning(warning_msg)

    # Handle errors
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        # Add helpful guidance
        error_msg += "\n\nTo fix configuration issues:"
        error_msg += "\n1. Update config.yaml with required settings"
        error_msg += (
            "\n2. Run in demo mode by omitting API keys for limited functionality"
        )

        if config.feature_flags.enable_error_handling:
            logging.error(error_msg)
            # In error handling mode, we still set the config but log the error
            # This allows the app to run with degraded functionality
        else:
            raise ValueError(error_msg)

    set_config(config)
    return config


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Set up logging based on configuration."""
    if config is None:
        config = get_config().logging

    # Configure root logger - handle both enum and string values
    level_value = config.level.value if hasattr(config.level, "value") else config.level
    logging.basicConfig(
        level=getattr(logging, level_value.upper()), format=config.format, force=True
    )

    # Add file handler if file path is specified
    if config.file_path:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
        )
        file_handler.setFormatter(logging.Formatter(config.format))

        # Handle both enum and string values for file handler level
        level_value = (
            config.level.value if hasattr(config.level, "value") else config.level
        )
        file_handler.setLevel(getattr(logging, level_value.upper()))

        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)


def get_config_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get configuration variable."""
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
