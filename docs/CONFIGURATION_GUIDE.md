# Configuration System Guide

This guide provides comprehensive documentation for the new centralized configuration system in the refactored Agentic Ticker codebase.

## Table of Contents

1. [Overview](#overview)
2. [Configuration Architecture](#configuration-architecture)
3. [Configuration Methods](#configuration-methods)
4. [Environment Variables](#environment-variables)
5. [Configuration Files](#configuration-files)
6. [Feature Flags](#feature-flags)
7. [API Configuration](#api-configuration)
8. [Analysis Parameters](#analysis-parameters)
9. [Logging Configuration](#logging-configuration)
10. [UI Configuration](#ui-configuration)
11. [Advanced Configuration](#advanced-configuration)
12. [Migration Guide](#migration-guide)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)

---

## Overview

The new configuration system provides:

- **Centralized Management**: Single source of truth for all settings
- **Multiple Input Methods**: Environment variables, files, programmatic access
- **Type Safety**: Pydantic models with validation
- **Hot Reload**: Configuration changes without restart
- **Backward Compatibility**: Existing environment variables continue to work
- **Comprehensive Validation**: Error detection and helpful messages

### Key Benefits

1. **Consistency**: All modules use the same configuration approach
2. **Flexibility**: Multiple ways to configure the system
3. **Maintainability**: Centralized configuration management
4. **Reliability**: Built-in validation and error handling
5. **Performance**: Optimized loading and caching

---

## Configuration Architecture

### Configuration Hierarchy

```
1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration files (JSON/YAML)
4. Default values (lowest priority)
```

### Core Configuration Classes

```python
from src.config import (
    AppConfig,           # Main configuration container
    GeminiConfig,        # Google Gemini API settings
    CoinGeckoConfig,     # CoinGecko API settings  
    YahooFinanceConfig,  # Yahoo Finance settings
    DDGConfig,          # DuckDuckGo search settings
    AnalysisConfig,     # Analysis parameters
    LoggingConfig,      # Logging configuration
    FeatureFlags,       # Feature toggle settings
    UIConfig            # User interface settings
)
```

---

## Configuration Methods

### Method 1: Environment Variables (Recommended for Deployment)

```bash
# Set environment variables
export GEMINI_API_KEY="your-actual-api-key"
export GEMINI_MODEL="gemini-2.5-flash-lite"
export LOG_LEVEL="INFO"
export ENABLE_WEB_SEARCH="true"
export DEFAULT_ANALYSIS_DAYS="30"

# Run application
python agentic_ticker.py
```

### Method 2: Configuration Files (Recommended for Development)

**JSON Configuration (`config.json`):**
```json
{
  "gemini": {
    "api_key": "your-api-key",
    "model": "gemini-2.5-flash-lite",
    "temperature": 0.2,
    "max_tokens": 8192
  },
  "analysis": {
    "default_days": 30,
    "default_threshold": 2.0,
    "default_forecast_days": 5
  },
  "feature_flags": {
    "enable_web_search": true,
    "enable_crypto_analysis": true,
    "enable_stock_analysis": true
  }
}
```

**YAML Configuration (`config.yaml`):**
```yaml
gemini:
  api_key: your-api-key
  model: gemini-2.5-flash-lite
  temperature: 0.2
  max_tokens: 8192

analysis:
  default_days: 30
  default_threshold: 2.0
  default_forecast_days: 5

feature_flags:
  enable_web_search: true
  enable_crypto_analysis: true
  enable_stock_analysis: true
```

### Method 3: Programmatic Configuration

```python
from src.config import load_config, AppConfig, GeminiConfig

# Load existing configuration
config = load_config()

# Modify configuration programmatically
config.gemini.api_key = "your-api-key"
config.analysis.default_days = 60
config.feature_flags.enable_web_search = False

# Save to file
config.save_to_file("custom_config.json")

# Create configuration from scratch
new_config = AppConfig()
new_config.gemini = GeminiConfig(
    api_key="your-api-key",
    model="gemini-2.5-flash-lite",
    temperature=0.1
)
```

---

## Environment Variables

### Core Environment Variables

```bash
# Gemini API Configuration
export GEMINI_API_KEY="your-actual-api-key"
export GEMINI_MODEL="gemini-2.5-flash-lite"
export GEMINI_API_BASE="https://generativelanguage.googleapis.com/v1beta"

# CoinGecko API Configuration
export COINGECKO_DEMO_API_KEY="your-demo-key"
export COINGECKO_API_KEY="your-pro-key"

# Logging Configuration
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Feature Flags
export ENABLE_WEB_SEARCH="true"
export ENABLE_CRYPTO_ANALYSIS="true"
export ENABLE_STOCK_ANALYSIS="true"
export ENABLE_FORECASTING="true"
export ENABLE_TECHNICAL_INDICATORS="true"
export ENABLE_ANIMATIONS="true"
export ENABLE_CACHING="true"
export ENABLE_RETRY_LOGIC="true"
export ENABLE_ERROR_HANDLING="true"
export ENABLE_VALIDATION="true"
```

### Advanced Environment Variables

```bash
# Compatibility Layer
export COMPATIBILITY_ENABLED="true"
export COMPATIBILITY_WARNINGS="true"
export COMPATIBILITY_STRICT="false"
export COMPATIBILITY_FALLBACK="true"
export COMPATIBILITY_DEADLINE="2025-12-31"

# Analysis Parameters
export DEFAULT_ANALYSIS_DAYS="30"
export DEFAULT_THRESHOLD="2.0"
export DEFAULT_FORECAST_DAYS="5"
export MAX_ANALYSIS_STEPS="10"

# API Timeouts
export GEMINI_TIMEOUT="120"
export COINGECKO_TIMEOUT="30"
export YAHOO_FINANCE_TIMEOUT="30"
export DDG_TIMEOUT="30"

# UI Configuration
export CHART_HEIGHT="500"
export ANIMATION_DURATION="500"
export TRANSITION_DURATION="300"
```

---

## Configuration Files

### Configuration File Locations

The system searches for configuration files in this order:

1. `config.json` (current directory)
2. `config.yaml` (current directory)
3. `.agentic-ticker.json` (current directory)
4. `.agentic-ticker.yaml` (current directory)
5. `~/.agentic-ticker.json` (home directory)
6. `~/.agentic-ticker.yaml` (home directory)

### Complete Configuration File Example

**`config.json`:**
```json
{
  "gemini": {
    "api_key": "your-api-key",
    "model": "gemini-2.5-flash-lite",
    "api_base": "https://generativelanguage.googleapis.com/v1beta",
    "temperature": 0.2,
    "max_tokens": 8192,
    "timeout": 120
  },
  "coingecko": {
    "demo_api_key": "your-demo-key",
    "pro_api_key": "your-pro-key",
    "environment": "demo",
    "timeout": 30
  },
  "yahoo_finance": {
    "timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 1.0
  },
  "ddg": {
    "max_results": 3,
    "region": "us-en",
    "safesearch": "moderate",
    "timeout": 30
  },
  "analysis": {
    "default_days": 30,
    "default_threshold": 2.0,
    "default_forecast_days": 5,
    "max_analysis_steps": 10,
    "min_data_points": 5,
    "volatility_window": 10,
    "ma5_window": 5,
    "ma10_window": 10
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": null,
    "max_file_size": 10485760,
    "backup_count": 5
  },
  "feature_flags": {
    "enable_web_search": true,
    "enable_crypto_analysis": true,
    "enable_stock_analysis": true,
    "enable_forecasting": true,
    "enable_technical_indicators": true,
    "enable_animations": true,
    "enable_caching": true,
    "enable_retry_logic": true,
    "enable_error_handling": true,
    "enable_validation": true
  },
  "ui": {
    "page_title": "Agentic-Ticker (Gemini)",
    "page_icon": "📈",
    "layout": "wide",
    "chart_height": 500,
    "animation_duration": 500,
    "transition_duration": 300
  },
  "config_file_path": null,
  "hot_reload_enabled": false,
  "hot_reload_interval": 60
}
```

---

## Feature Flags

### Available Feature Flags

| Flag | Description | Default | Impact |
|------|-------------|---------|---------|
| `enable_web_search` | Enable web search functionality | `true` | Disables DDGS search if false |
| `enable_crypto_analysis` | Enable cryptocurrency analysis | `true` | Disables crypto features if false |
| `enable_stock_analysis` | Enable stock analysis | `true` | Disables stock features if false |
| `enable_forecasting` | Enable price forecasting | `true` | Disables forecast generation if false |
| `enable_technical_indicators` | Enable technical indicators | `true` | Disables indicator calculation if false |
| `enable_animations` | Enable chart animations | `true` | Disables animations for better performance |
| `enable_caching` | Enable function caching | `true` | Disables all caching if false |
| `enable_retry_logic` | Enable retry mechanisms | `true` | Disables automatic retries if false |
| `enable_error_handling` | Enable error handling | `true` | Disables error handling decorators if false |
| `enable_validation` | Enable input validation | `true` | Disables input validation if false |

### Feature Flag Configuration

```python
from src.config import load_config

config = load_config()

# Disable features for testing
config.feature_flags.enable_animations = False
config.feature_flags.enable_caching = False

# Enable all features for production
config.feature_flags.enable_web_search = True
config.feature_flags.enable_crypto_analysis = True
config.feature_flags.enable_stock_analysis = True

# Save configuration
config.save_to_file("production_config.json")
```

### Environment Variable Feature Flags

```bash
# Disable animations for better performance
export ENABLE_ANIMATIONS="false"

# Enable all analysis features
export ENABLE_CRYPTO_ANALYSIS="true"
export ENABLE_STOCK_ANALYSIS="true"
export ENABLE_FORECASTING="true"

# Disable caching for development
export ENABLE_CACHING="false"

# Enable strict validation
export ENABLE_VALIDATION="true"
```

---

## API Configuration

### Gemini API Configuration

```python
from src.config import load_config

config = load_config()

# Configure Gemini API
config.gemini.api_key = "your-actual-api-key"
config.gemini.model = "gemini-2.5-flash-lite"
config.gemini.temperature = 0.2  # 0.0-2.0
config.gemini.max_tokens = 8192
config.gemini.timeout = 120  # seconds

# Environment variables
export GEMINI_API_KEY="your-api-key"
export GEMINI_MODEL="gemini-2.5-flash-lite"
export GEMINI_API_BASE="https://generativelanguage.googleapis.com/v1beta"
```

### CoinGecko API Configuration

```python
# Configure CoinGecko API
config.coingecko.demo_api_key = "your-demo-key"
config.coingecko.pro_api_key = "your-pro-key"
config.coingecko.environment = "demo"  # or "pro"
config.coingecko.timeout = 30  # seconds

# Environment variables
export COINGECKO_DEMO_API_KEY="your-demo-key"
export COINGECKO_API_KEY="your-pro-key"
```

### Yahoo Finance Configuration

```python
# Configure Yahoo Finance
config.yahoo_finance.timeout = 30
config.yahoo_finance.retry_attempts = 3
config.yahoo_finance.retry_delay = 1.0  # seconds
```

### DuckDuckGo Search Configuration

```python
# Configure DDG search
config.ddg.max_results = 3
config.ddg.region = "us-en"
config.ddg.safesearch = "moderate"  # or "strict", "off"
config.ddg.timeout = 30  # seconds
```

---

## Analysis Parameters

### Default Analysis Settings

```python
from src.config import load_config

config = load_config()

# Configure analysis defaults
config.analysis.default_days = 30  # Historical data period
config.analysis.default_threshold = 2.0  # Event detection threshold
config.analysis.default_forecast_days = 5  # Forecast horizon
config.analysis.max_analysis_steps = 10  # Maximum LLM iterations
config.analysis.min_data_points = 5  # Minimum required data points

# Technical indicator settings
config.analysis.volatility_window = 10
config.analysis.ma5_window = 5  # 5-day moving average
config.analysis.ma10_window = 10  # 10-day moving average
```

### Custom Analysis Configuration

```json
{
  "analysis": {
    "default_days": 60,
    "default_threshold": 1.5,
    "default_forecast_days": 10,
    "max_analysis_steps": 15,
    "min_data_points": 10,
    "volatility_window": 20,
    "ma5_window": 5,
    "ma10_window": 10,
    "ma20_window": 20,
    "rsi_window": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9
  }
}
```

---

## Logging Configuration

### Basic Logging Setup

```python
from src.config import load_config, setup_logging

config = load_config()

# Configure logging
config.logging.level = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
config.logging.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Set up logging
setup_logging(config)
```

### Advanced Logging Configuration

```json
{
  "logging": {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    "file_path": "/var/log/agentic-ticker.log",
    "max_file_size": 10485760,
    "backup_count": 5
  }
}
```

### Environment Variable Logging

```bash
# Set log level
export LOG_LEVEL="DEBUG"

# Enable file logging
export LOG_FILE_PATH="/path/to/log/file.log"

# Configure log format
export LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## UI Configuration

### User Interface Settings

```python
from src.config import load_config

config = load_config()

# Configure UI appearance
config.ui.page_title = "My Stock Analyzer"
config.ui.page_icon = "📊"
config.ui.layout = "wide"  # or "centered"
config.ui.chart_height = 600
config.ui.animation_duration = 750
config.ui.transition_duration = 400
```

### UI Configuration File

```json
{
  "ui": {
    "page_title": "Agentic-Ticker Pro",
    "page_icon": "💰",
    "layout": "wide",
    "chart_height": 700,
    "animation_duration": 1000,
    "transition_duration": 500,
    "theme": {
      "primary_color": "#1f77b4",
      "secondary_color": "#ff7f0e",
      "background_color": "#ffffff",
      "text_color": "#333333"
    }
  }
}
```

---

## Advanced Configuration

### Hot Reload Configuration

```python
from src.config import load_config

config = load_config()

# Enable hot reload for development
config.hot_reload_enabled = True
config.hot_reload_interval = 30  # seconds

# Save configuration
config.save_to_file("development_config.json")

# Manual reload
from src.config import reload_config
reload_config()
```

### Configuration Validation

```python
from src.config import load_config

config = load_config()

# Validate configuration
errors = config.validate()

if errors:
    print("Configuration validation failed:")
    for error in errors:
        print(f"  - {error}")
    raise ValueError("Invalid configuration")
else:
    print("Configuration is valid!")
```

### Environment Variables Export

```python
from src.config import load_config

config = load_config()

# Get environment variables representation
env_vars = config.get_env_vars()

# Create shell script
with open("set_env_vars.sh", "w") as f:
    f.write("#!/bin/bash\n")
    for key, value in env_vars.items():
        f.write(f"export {key}=\"{value}\"\n")

# Make executable
import os
os.chmod("set_env_vars.sh", 0o755)
```

---

## Migration Guide

### From Environment Variables to Configuration Files

**Before (Environment Variables Only):**
```bash
export GEMINI_API_KEY="old-key"
export DEFAULT_DAYS="30"
export ENABLE_WEB_SEARCH="true"
```

**After (Configuration File):**
```json
{
  "gemini": {
    "api_key": "new-key"
  },
  "analysis": {
    "default_days": 30
  },
  "feature_flags": {
    "enable_web_search": true
  }
}
```

### From Hard-coded Values to Configuration

**Before (Hard-coded):**
```python
# Old approach - hard-coded values
API_KEY = "old-api-key"
DEFAULT_DAYS = 30
ENABLE_SEARCH = True
```

**After (Configuration):**
```python
# New approach - centralized configuration
from src.config import load_config

config = load_config()
api_key = config.gemini.api_key
default_days = config.analysis.default_days
enable_search = config.feature_flags.enable_web_search
```

### Gradual Migration Strategy

```python
# Step 1: Load configuration alongside existing code
from src.config import load_config
config = load_config()

# Step 2: Replace hard-coded values gradually
# Old: API_KEY = "hard-coded-key"
# New: API_KEY = config.gemini.api_key

# Step 3: Remove old configuration code
# Delete hard-coded values and old config files

# Step 4: Use configuration throughout codebase
# All settings now come from centralized configuration
```

---

## Best Practices

### 1. Use Environment Variables for Sensitive Data

```bash
# Good: API keys in environment variables
export GEMINI_API_KEY="your-secret-key"

# Avoid: API keys in configuration files
# {
#   "gemini": {
#     "api_key": "your-secret-key"  # Don't do this!
#   }
# }
```

### 2. Use Configuration Files for Complex Settings

```json
// Good: Complex settings in configuration files
{
  "analysis": {
    "technical_indicators": {
      "rsi_period": 14,
      "macd_fast": 12,
      "macd_slow": 26,
      "bollinger_bands_period": 20,
      "bollinger_bands_std": 2
    }
  }
}
```

### 3. Validate Configuration Early

```python
from src.config import load_config

def initialize_application():
    config = load_config()
    
    # Validate before using
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {errors}")
    
    # Now safe to use
    setup_logging(config)
    return config
```

### 4. Use Feature Flags for Gradual Rollouts

```python
from src.config import load_config

config = load_config()

# Gradual feature rollout
if config.feature_flags.enable_new_feature:
    # New feature code
    process_with_new_algorithm()
else:
    # Legacy code
    process_with_legacy_algorithm()
```

### 5. Document Configuration Changes

```python
# Document configuration changes
config_changes = {
    "version": "2.0",
    "changes": [
        "Added enable_animations feature flag",
        "Changed default analysis days from 30 to 60",
        "Added new technical indicators configuration"
    ]
}
```

---

## Troubleshooting

### Issue: Configuration Not Loading

**Symptoms:**
- Default values being used instead of configured values
- Configuration file not found

**Solutions:**
```python
from src.config import load_config
import os

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Check for configuration files
config_files = ['config.json', 'config.yaml', '.agentic-ticker.json', '.agentic-ticker.yaml']
for file in config_files:
    exists = os.path.exists(file)
    print(f"{file}: {'EXISTS' if exists else 'NOT FOUND'}")

# Load with explicit path
config = load_config("/path/to/config.json")

# Check what was loaded
print(f"Config file used: {config.config_file_path}")
print(f"Gemini API key set: {bool(config.gemini.api_key)}")
```

### Issue: Configuration Validation Failing

**Symptoms:**
- Validation errors when loading configuration
- Application failing to start

**Solutions:**
```python
from src.config import load_config

config = load_config()

# Get validation errors
errors = config.validate()

if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
    
    # Fix common issues
    if "GEMINI_API_KEY is required" in errors:
        print("Set GEMINI_API_KEY environment variable")
    
    if "temperature must be between 0 and 2" in errors:
        print("Set temperature to value between 0.0 and 2.0")
```

### Issue: Environment Variables Not Working

**Symptoms:**
- Environment variables not being recognized
- Configuration values not updating

**Solutions:**
```python
import os
from src.config import load_config

# Check environment variables
env_vars = [
    'GEMINI_API_KEY', 'GEMINI_MODEL', 'LOG_LEVEL',
    'ENABLE_WEB_SEARCH', 'ENABLE_CRYPTO_ANALYSIS'
]

for var in env_vars:
    value = os.getenv(var)
    print(f"{var}: {value if value else 'NOT SET'}")

# Force reload configuration
config = load_config()  # Reloads from environment

# Check if values are being used
print(f"Gemini model: {config.gemini.model}")
print(f"Web search enabled: {config.feature_flags.enable_web_search}")
```

### Issue: Hot Reload Not Working

**Symptoms:**
- Configuration changes not reflected
- Hot reload interval not respected

**Solutions:**
```python
from src.config import load_config

config = load_config()

# Enable hot reload
config.hot_reload_enabled = True
config.hot_reload_interval = 30  # seconds

# Manual reload
from src.config import reload_config
reload_config()

# Check if hot reload is working
import time
print(f"Current config: {config.gemini.model}")
time.sleep(35)  # Wait for reload interval
reload_config()
print(f"Updated config: {config.gemini.model}")
```

---

## Configuration Reference

### Complete Configuration Schema

```python
# Main configuration class
@dataclass
class AppConfig:
    gemini: GeminiConfig
    coingecko: CoinGeckoConfig
    yahoo_finance: YahooFinanceConfig
    ddg: DDGConfig
    analysis: AnalysisConfig
    logging: LoggingConfig
    feature_flags: FeatureFlags
    ui: UIConfig
    config_file_path: Optional[str]
    hot_reload_enabled: bool
    hot_reload_interval: int
```

### Configuration Validation Rules

| Field | Validation Rule | Error Message |
|-------|----------------|---------------|
| `gemini.api_key` | Required if using Gemini | "GEMINI_API_KEY is required" |
| `gemini.temperature` | 0.0-2.0 range | "Gemini temperature must be between 0 and 2" |
| `gemini.timeout` | Positive integer | "Gemini timeout must be positive" |
| `analysis.default_days` | Positive integer | "Default days must be positive" |
| `analysis.default_threshold` | Positive float | "Default threshold must be positive" |
| `logging.file_path` | Writable directory | "Cannot create log directory" |

This comprehensive configuration guide should help you effectively use and manage the new centralized configuration system in the refactored Agentic Ticker codebase.