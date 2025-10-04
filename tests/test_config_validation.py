"""
Test configuration validation system.
"""

import pytest
import os
import tempfile
from pathlib import Path

# Add src to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import from src.config directly
from src.config import AppConfig, GeminiConfig


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_default_config_validation(self):
        """Test that default configuration passes validation."""
        # Create config without loading from file
        config = AppConfig()
        config.config_file_path = None  # Don't load from file

        # Manually clear API keys to test warning behavior
        original_gemini_key = config.gemini.api_key
        original_coingecko_key = config.coingecko.demo_api_key

        try:
            config.gemini.api_key = ""
            config.coingecko.demo_api_key = ""
            config.coingecko.pro_api_key = ""

            errors, warnings = config.validate()

            # Should have no errors, only warnings about missing API keys
            assert len(errors) == 0
            assert len(warnings) > 0  # Should warn about missing API keys
            assert any("GEMINI_API_KEY" in w for w in warnings)
        finally:
            # Restore original values
            config.gemini.api_key = original_gemini_key
            config.coingecko.demo_api_key = original_coingecko_key

    def test_invalid_gemini_temperature(self):
        """Test validation of invalid Gemini temperature."""
        config = AppConfig()
        config.gemini.temperature = 3.0  # Invalid: > 2

        errors, warnings = config.validate()
        assert any("temperature must be between 0 and 2" in e for e in errors)

        config.gemini.temperature = -1.0  # Invalid: < 0
        errors, warnings = config.validate()
        assert any("temperature must be between 0 and 2" in e for e in errors)

    def test_invalid_timeout_values(self):
        """Test validation of invalid timeout values."""
        config = AppConfig()

        # Test Gemini timeout
        config.gemini.timeout = 0
        errors, warnings = config.validate()
        assert any("Gemini timeout must be positive" in e for e in errors)

        # Test CoinGecko timeout
        config.gemini.timeout = 120  # Reset to valid
        config.coingecko.timeout = -5
        errors, warnings = config.validate()
        assert any("CoinGecko timeout must be positive" in e for e in errors)

    def test_invalid_analysis_values(self):
        """Test validation of invalid analysis configuration."""
        config = AppConfig()

        # Test negative values
        config.analysis.default_days = -1
        errors, warnings = config.validate()
        assert any("Default days must be positive" in e for e in errors)

        config.analysis.default_days = 30  # Reset
        config.analysis.default_threshold = 0
        errors, warnings = config.validate()
        assert any("Default threshold must be positive" in e for e in errors)

    def test_invalid_url_validation(self):
        """Test URL validation."""
        config = AppConfig()

        # Test invalid URL
        config.gemini.api_base = "not-a-url"
        errors, warnings = config.validate()
        assert any("API base URL is invalid" in e for e in errors)

        # Test valid URLs
        valid_urls = [
            "https://api.example.com",
            "http://localhost:8080",
            "https://generativelanguage.googleapis.com/v1beta",
        ]

        for url in valid_urls:
            config.gemini.api_base = url
            errors, warnings = config.validate()
            assert not any("API base URL is invalid" in e for e in errors)

    def test_logging_directory_creation(self):
        """Test logging directory creation validation."""
        config = AppConfig()

        # Test with non-existent directory that can't be created
        config.logging.file_path = "/invalid/path/that/cannot/be/created/test.log"
        errors, warnings = config.validate()
        assert any("Cannot create log directory" in e for e in errors)

    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        config = AppConfig()
        config.feature_flags.enable_validation = False
        config.gemini.temperature = 999  # Invalid value

        errors, warnings = config.validate()
        assert len(errors) == 0  # Should not validate when disabled
        assert len(warnings) == 0

    def test_runtime_validation(self):
        """Test runtime configuration change validation."""
        config = AppConfig()

        # Test valid change
        valid, msg = config.validate_runtime_change("gemini", "temperature", 1.0)
        assert valid is True
        assert msg == "Valid"

        # Test invalid change
        valid, msg = config.validate_runtime_change("gemini", "temperature", 5.0)
        assert valid is False
        assert "temperature must be between 0 and 2" in msg

        # Test invalid type
        valid, msg = config.validate_runtime_change(
            "analysis", "default_days", "not-a-number"
        )
        assert valid is False
        assert "Invalid value type" in msg

    def test_mode_info(self):
        """Test mode information detection."""
        config = AppConfig()

        # Store original keys
        original_gemini_key = config.gemini.api_key
        original_coingecko_demo = config.coingecko.demo_api_key
        original_coingecko_pro = config.coingecko.pro_api_key

        try:
            # Test demo mode (no API keys)
            config.gemini.api_key = ""
            config.coingecko.demo_api_key = ""
            config.coingecko.pro_api_key = ""

            mode_info = config.get_mode_info()
            assert mode_info["mode"] == "demo"
            assert mode_info["has_gemini_key"] is False
            assert mode_info["has_coingecko_key"] is False
            assert "AI Analysis (mock data)" in mode_info["limited_features"]

            # Test partial mode (only Gemini)
            config.gemini.api_key = "test-key"
            mode_info = config.get_mode_info()
            assert mode_info["mode"] == "partial"
            assert mode_info["has_gemini_key"] is True
            assert "AI Analysis" in mode_info["available_features"]

            # Test full mode (both keys)
            config.coingecko.pro_api_key = "test-coingecko-key"
            mode_info = config.get_mode_info()
            assert mode_info["mode"] == "full"
            assert mode_info["has_coingecko_key"] is True
        finally:
            # Restore original values
            config.gemini.api_key = original_gemini_key
            config.coingecko.demo_api_key = original_coingecko_demo
            config.coingecko.pro_api_key = original_coingecko_pro

    def test_config_file_loading(self):
        """Test loading configuration from file."""
        # Create a temporary config file
        config_data = {
            "gemini": {"temperature": 1.5, "timeout": 60},
            "analysis": {"default_days": 20},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(config_data, f)
            temp_file = f.name

        try:
            config = AppConfig(config_file_path=temp_file)
            assert config.gemini.temperature == 1.5
            assert config.gemini.timeout == 60
            assert config.analysis.default_days == 20
        finally:
            os.unlink(temp_file)

    def test_config_yaml_loading(self):
        """Test that configuration loads from config.yaml."""
        # Test that config loads from YAML file instead of environment
        gemini_config = GeminiConfig()
        # Should use default values from config.yaml, not environment
        assert gemini_config.api_key == "" or gemini_config.api_key is None

    def test_warning_vs_error_distinction(self):
        """Test that warnings and errors are properly distinguished."""
        config = AppConfig()

        # Store original keys
        original_gemini_key = config.gemini.api_key
        original_coingecko_demo = config.coingecko.demo_api_key
        original_coingecko_pro = config.coingecko.pro_api_key

        try:
            # Clear API keys to generate warnings
            config.gemini.api_key = ""
            config.coingecko.demo_api_key = ""
            config.coingecko.pro_api_key = ""

            # Should only have warnings (missing API keys)
            errors, warnings = config.validate()
            assert len(errors) == 0
            assert len(warnings) > 0

            # Add an error condition
            config.gemini.temperature = -1
            errors, warnings = config.validate()
            assert len(errors) > 0
            assert len(warnings) > 0

            # Check that error messages are different from warning messages
            error_messages = " ".join(errors)
            warning_messages = " ".join(warnings)
            assert "temperature" in error_messages
            assert "GEMINI_API_KEY" in warning_messages
        finally:
            # Restore original values
            config.gemini.api_key = original_gemini_key
            config.coingecko.demo_api_key = original_coingecko_demo
            config.coingecko.pro_api_key = original_coingecko_pro


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
