import pytest
from src.data_models import FunctionCall, PlannerJSON


class TestFunctionCall:
    def test_function_call_with_args(self):
        """Test FunctionCall model with arguments"""
        func_call = FunctionCall(
            name="test_function",
            args={"param1": "value1", "param2": 42}
        )
        
        assert func_call.name == "test_function"
        assert func_call.args == {"param1": "value1", "param2": 42}
    
    def test_function_call_without_args(self):
        """Test FunctionCall model without arguments"""
        func_call = FunctionCall(name="test_function")
        
        assert func_call.name == "test_function"
        assert func_call.args is None
    
    def test_function_call_with_empty_args(self):
        """Test FunctionCall model with empty arguments"""
        func_call = FunctionCall(name="test_function", args={})
        
        assert func_call.name == "test_function"
        assert func_call.args == {}


class TestPlannerJSON:
    def test_planner_json_with_call(self):
        """Test PlannerJSON model with function call"""
        func_call = FunctionCall(name="test_function", args={"param": "value"})
        planner_json = PlannerJSON(call=func_call)
        
        assert planner_json.call == func_call
        assert planner_json.final is None
    
    def test_planner_json_with_final(self):
        """Test PlannerJSON model with final response"""
        planner_json = PlannerJSON(final="This is the final response")
        
        assert planner_json.final == "This is the final response"
        assert planner_json.call is None
    
    def test_planner_json_with_both(self):
        """Test PlannerJSON model with both call and final"""
        func_call = FunctionCall(name="test_function")
        planner_json = PlannerJSON(
            call=func_call,
            final="Final response"
        )
        
        assert planner_json.call == func_call
        assert planner_json.final == "Final response"
    
    def test_planner_json_empty(self):
        """Test PlannerJSON model with no data"""
        planner_json = PlannerJSON()
        
        assert planner_json.call is None
        assert planner_json.final is None


if __name__ == "__main__":
    pytest.main([__file__])