import pytest
from unittest.mock import Mock, patch
from src.orchestrator import Orchestrator


class TestIntegration:
    """Integration tests for the agentic ticker system"""
    
    def test_orchestrator_tools_integration(self):
        """Test that orchestrator can access all required tools"""
        orchestrator = Orchestrator()
        
        # Test that all expected tools are available
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
            assert callable(orchestrator.tools[tool_name])
    
    def test_tools_spec_integration(self):
        """Test tools spec generation for LLM integration"""
        orchestrator = Orchestrator()
        spec = orchestrator.tools_spec()
        
        # Test that spec is properly formatted
        assert isinstance(spec, list)
        assert len(spec) == len(orchestrator.tools)
        
        # Test each tool spec has required fields
        for tool_spec in spec:
            assert 'name' in tool_spec
            assert 'docstring' in tool_spec
            assert 'signature' in tool_spec
            assert isinstance(tool_spec['name'], str)
            assert isinstance(tool_spec['docstring'], str)
            assert isinstance(tool_spec['signature'], str)
    
    @patch('src.services.ddgs_search')
    def test_search_function_integration(self, mock_search):
        """Test that search function integrates properly"""
        mock_search.return_value = [
            {'title': 'Test Result', 'href': 'http://example.com', 'content': 'Test content'}
        ]
        
        from src.services import ddgs_search
        result = ddgs_search("test query")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['title'] == 'Test Result'
        mock_search.assert_called_once_with("test query")
    
    def test_data_models_integration(self):
        """Test that data models work with the system"""
        from src.data_models import FunctionCall, PlannerJSON
        
        # Test FunctionCall model
        func_call = FunctionCall(name="test_function", args={"param": "value"})
        assert func_call.name == "test_function"
        assert func_call.args == {"param": "value"}
        
        # Test PlannerJSON model
        planner_json = PlannerJSON(call=func_call)
        assert planner_json.call == func_call
        assert planner_json.final is None
    
    def test_error_sanitization_integration(self):
        """Test error message sanitization in orchestrator"""
        orchestrator = Orchestrator()
        
        # Test with API key
        error_msg = "Error: https://api.example.com?key=secret123&param=value failed"
        sanitized = orchestrator._sanitize_error_message(error_msg)
        
        assert 'secret123' not in sanitized
        assert 'key=' in sanitized  # Should keep parameter name
        assert 'api.example.com' in sanitized  # Should keep domain
    
    @patch('src.services.validate_ticker')
    def test_validation_function_integration(self, mock_validate):
        """Test ticker validation integration"""
        mock_validate.return_value = "AAPL"
        
        from src.services import validate_ticker
        result = validate_ticker("Apple")
        
        assert result == "AAPL"
        mock_validate.assert_called_once_with("Apple")


if __name__ == "__main__":
    pytest.main([__file__])