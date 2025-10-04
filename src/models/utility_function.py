from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from src.decorators import handle_errors, log_execution, validate_inputs
from src.exceptions import ValidationError, ParameterError, UtilityFunctionError
from src.models.function_parameter import FunctionParameter


class UtilityFunction(BaseModel):
    """Represents a utility function with metadata for code refactoring."""

    name: str = Field(
        ...,
        description="Function name in snake_case format",
        pattern=r"^[a-z][a-z0-9_]*[a-z0-9]$",
    )
    description: str = Field(..., description="Purpose and behavior of the function")
    parameters: List[FunctionParameter] = Field(
        default_factory=list, description="List of function parameters"
    )
    return_type: str = Field(..., description="Expected return type")
    lines_of_code: int = Field(
        ..., description="Number of lines of code in the function", gt=0
    )
    complexity_score: Optional[float] = Field(
        None, description="Optional complexity score for code analysis", ge=0.0
    )
    is_decorated: bool = Field(
        False, description="Whether the function has decorators applied"
    )
    decorators: List[str] = Field(
        default_factory=list,
        description="List of decorator names applied to the function",
    )

    @field_validator("name")
    @classmethod
    def validate_snake_case(cls, v):
        """Validate that the function name follows snake_case convention."""
        if not v:
            raise ValidationError("Function name cannot be empty")
        if not v.islower():
            raise ValidationError("Function name must be in snake_case (lowercase)")
        if not v.replace("_", "").isalnum():
            raise ValidationError(
                "Function name can only contain lowercase letters, numbers, and "
                "underscores"
            )
        if v.startswith("_") or v.endswith("_"):
            raise ValidationError("Function name cannot start or end with underscore")
        return v

    @field_validator("decorators")
    @classmethod
    def validate_decorators(cls, v, info):
        """Validate decorators list consistency with is_decorated flag."""
        is_decorated = info.data.get("is_decorated", False)
        if v and not is_decorated:
            raise ValidationError(
                "is_decorated must be True when decorators list is not empty"
            )
        if is_decorated and not v:
            raise ValidationError(
                "decorators list cannot be empty when is_decorated is True"
            )
        return v

    @field_validator("lines_of_code")
    @classmethod
    def validate_lines_of_code(cls, v):
        """Validate that lines of code is a positive integer."""
        if v <= 0:
            raise ValidationError("lines_of_code must be a positive integer")
        return v

    @handle_errors(log_errors=True, reraise_exceptions=UtilityFunctionError)
    @log_execution(include_args=False, include_result=False)
    def add_parameter(self, parameter: FunctionParameter) -> None:
        """Add a parameter to the function."""
        if not parameter:
            raise ParameterError("Parameter cannot be None or empty")
        if any(p.name == parameter.name for p in self.parameters):
            raise ParameterError(f"Parameter '{parameter.name}' already exists")

        self.parameters.append(parameter)

    @handle_errors(log_errors=True, reraise_exceptions=UtilityFunctionError)
    @log_execution(include_args=False, include_result=False)
    @validate_inputs(parameter_name="non_empty_string")
    def remove_parameter(self, parameter_name: str) -> bool:
        """Remove a parameter by name."""
        if not any(p.name == parameter_name for p in self.parameters):
            raise ParameterError(f"Parameter '{parameter_name}' not found")

        original_count = len(self.parameters)
        self.parameters = [p for p in self.parameters if p.name != parameter_name]
        return len(self.parameters) < original_count

    @handle_errors(log_errors=True, reraise_exceptions=UtilityFunctionError)
    @log_execution(include_args=False, include_result=False)
    @validate_inputs(decorator_name="non_empty_string")
    def add_decorator(self, decorator_name: str) -> None:
        """Add a decorator to the function."""
        if decorator_name in self.decorators:
            raise ValidationError(f"Decorator '{decorator_name}' already exists")

        self.decorators.append(decorator_name)
        self.is_decorated = True

    @handle_errors(log_errors=True, reraise_exceptions=UtilityFunctionError)
    @log_execution(include_args=False, include_result=False)
    @validate_inputs(decorator_name="non_empty_string")
    def remove_decorator(self, decorator_name: str) -> bool:
        """Remove a decorator from the function."""
        if decorator_name not in self.decorators:
            raise ValidationError(f"Decorator '{decorator_name}' not found")

        self.decorators.remove(decorator_name)
        if not self.decorators:
            self.is_decorated = False
        return True

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "name": "calculate_api_metrics",
                "description": "Calculates performance metrics for API calls",
                "parameters": [
                    {"name": "response_time", "type": "float", "required": True},
                    {"name": "status_code", "type": "int", "required": True},
                ],
                "return_type": "Dict[str, float]",
                "lines_of_code": 25,
                "complexity_score": 3.2,
                "is_decorated": True,
                "decorators": ["handle_api_errors", "with_timing"],
            }
        }
