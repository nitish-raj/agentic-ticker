from typing import List
from enum import Enum
from pydantic import BaseModel, Field


class DecoratorConcern(str, Enum):
    """Enum representing different concerns that decorators can address."""

    ERROR_HANDLING = "error_handling"
    LOGGING = "logging"
    CACHING = "caching"
    VALIDATION = "validation"
    RETRY = "retry"
    TIMING = "timing"
    SECURITY = "security"


class ImplementationPattern(str, Enum):
    """Enum representing different implementation patterns for decorators."""

    FUNCTION_WRAPPER = "function_wrapper"
    CLASS_DECORATOR = "class_decorator"
    PROPERTY_DECORATOR = "property_decorator"
    METHOD_DECORATOR = "method_decorator"


class Decorator(BaseModel):
    """Pydantic model representing a decorator with its properties and metadata."""

    name: str = Field(..., description="Decorator name in snake_case format")

    description: str = Field(
        ..., description="Description of what the decorator does and its purpose"
    )

    concern: DecoratorConcern = Field(
        ..., description="The primary concern this decorator addresses"
    )

    implementation_pattern: ImplementationPattern = Field(
        ..., description="The implementation pattern used by this decorator"
    )

    target_functions: List[str] = Field(
        default_factory=list,
        description="List of function names this decorator can be applied to",
    )

    code_pattern: str = Field(
        ..., description="The actual decorator code pattern/template"
    )

    is_active: bool = Field(
        default=True,
        description="Whether this decorator is currently active and available for use",
    )

    usage_count: int = Field(
        default=0, description="Number of times this decorator has been used", ge=0
    )

    class Config:
        """Pydantic model configuration."""

        use_enum_values = True
        schema_extra = {
            "example": {
                "name": "log_execution",
                "description": "Logs function execution time and arguments",
                "concern": "logging",
                "implementation_pattern": "function_wrapper",
                "target_functions": ["process_data", "calculate_metrics"],
                "code_pattern": "def log_execution(func):\n"
                "    def wrapper(*args, **kwargs):\n"
                "        start_time = time.time()\n"
                "        result = func(*args, **kwargs)\n"
                "        end_time = time.time()\n"
                "        logger.info(f'{func.__name__} executed in "
                "{end_time - start_time:.2f}s')\n"
                "        return result\n"
                "    return wrapper",
                "is_active": True,
                "usage_count": 15,
            }
        }
