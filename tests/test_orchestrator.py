import pytest
from unittest.mock import Mock, patch
from src.orchestrator import Orchestrator
from src.planner import GeminiPlanner


class TestOrchestrator:
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly"""
        orchestrator = Orchestrator()
        
        assert hasattr(orchestrator, 'planner')
        assert hasattr(orchestrator, 'tools')
        assert isinstance(orchestrator.tools, dict)
        assert len(orchestrator.tools) > 0
    
    def test_tools_spec(self):
        """Test tools specification generation"""
        orchestrator = Orchestrator()
        spec = orchestrator.tools_spec()
        
        assert isinstance(spec, list)
        assert len(spec) > 0
        
        for tool in spec:
            assert 'name' in tool
            assert 'docstring' in tool
            assert 'signature' in tool
    
    def test_tools_contain_expected_functions(self):
        """Test that expected functions are in tools registry"""
        orchestrator = Orchestrator()
        
        expected_tools = [
            'validate_ticker',
            'get_company_info',
            'get_crypto_info',
            'load_prices',
            'load_crypto_prices',
            'compute_indicators',
            'detect_events',
            'forecast_prices',
            'build_report',
            'ddgs_search'
        ]
        
        for tool_name in expected_tools:
            assert tool_name in orchestrator.tools
    
    @patch('src.orchestrator.GeminiPlanner')
    def test_planner_initialization(self, mock_planner_class):
        """Test that planner is initialized correctly"""
        mock_planner_instance = Mock()
        mock_planner_class.return_value = mock_planner_instance
        
        orchestrator = Orchestrator()
        
        mock_planner_class.assert_called_once()
        assert orchestrator.planner == mock_planner_instance
    
    def test_sanitize_error_message(self):
        """Test error message sanitization"""
        orchestrator = Orchestrator()
        
        # Test with API key in error message
        error_with_key = "Request failed: https://api.example.com?key=secret123&param=value"
        sanitized = orchestrator._sanitize_error_message(error_with_key)
        
        assert 'secret123' not in sanitized
        assert 'key=' in sanitized  # Should keep the parameter name
        
        # Test with normal error message
        normal_error = "Network timeout occurred"
        sanitized_normal = orchestrator._sanitize_error_message(normal_error)
        
        assert sanitized_normal == normal_error


if __name__ == "__main__":
    pytest.main([__file__])