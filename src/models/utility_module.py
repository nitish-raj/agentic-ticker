import re
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field, validator

from src.decorators import handle_errors, log_execution, validate_inputs
from src.exceptions import (
    ValidationError, FunctionNotFoundError,
    DependencyError, FilePathError, UtilityModuleError
)
from src.models.utility_function import UtilityFunction


class UtilityModule(BaseModel):
    """Represents a utility module with its functions and metadata."""
    name: str = Field(..., description="Name of the utility module")
    description: str = Field(..., description="Purpose and scope of the module")
    file_path: str = Field(
        ..., description="Relative path where the module will be created"
    )
    functions: List[UtilityFunction] = Field(
        default_factory=list, description="List of utility functions in the module"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="External dependencies required by the module"
    )
    lines_saved: int = Field(
        0, description="Estimated lines of code saved", ge=0
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the module was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the module was last updated"
    )

    @validator('name')
    def validate_name(cls, v):
        if not v:
            raise ValidationError("Module name cannot be empty")
        if not re.match(r'^[a-z][a-z0-9_]*$', v):
            raise ValidationError("Module name must be in snake_case format")
        return v

    @validator('file_path')
    def validate_file_path(cls, v):
        if not v.endswith('.py'):
            raise FilePathError("File path must end with .py")
        if not v.startswith('src/'):
            raise FilePathError("File path must start with src/")
        if not re.match(r'^src/[a-z][a-z0-9_/]*\.py$', v):
            raise FilePathError(
                "File path must follow pattern: src/[a-z][a-z0-9_/]*.py"
            )
        return v

    @validator('updated_at', always=True)
    def validate_updated_at(cls, v, values):
        # Ensure updated_at is always set to current time when model is created/updated
        return datetime.utcnow()

    @property
    def total_lines_saved(self) -> int:
        """Calculate total lines saved across all functions in the module."""
        return sum(func.lines_of_code for func in self.functions)

    @property
    def function_count(self) -> int:
        """Get the number of functions in the module."""
        return len(self.functions)

    @handle_errors(log_errors=True, reraise_exceptions=UtilityModuleError)
    @log_execution(include_args=False, include_result=False)
    @validate_inputs(function='utility_function')
    def add_function(self, function: UtilityFunction) -> None:
        """Add a function to the module and update lines_saved."""
        if not function:
            raise ValidationError("Function cannot be None or empty")
        if any(f.name == function.name for f in self.functions):
            raise ValidationError(
                f"Function '{function.name}' already exists in module"
            )

        self.functions.append(function)
        self.lines_saved = self.total_lines_saved
        self.updated_at = datetime.utcnow()

    @handle_errors(log_errors=True, reraise_exceptions=UtilityModuleError)
    @log_execution(include_args=False, include_result=False)
    @validate_inputs(function_name='non_empty_string')
    def remove_function(self, function_name: str) -> bool:
        """Remove a function by name and update lines_saved."""
        if not any(f.name == function_name for f in self.functions):
            raise FunctionNotFoundError(
                f"Function '{function_name}' not found in module"
            )

        original_count = len(self.functions)
        self.functions = [f for f in self.functions if f.name != function_name]
        if len(self.functions) < original_count:
            self.lines_saved = self.total_lines_saved
            self.updated_at = datetime.utcnow()
            return True
        return False

    @handle_errors(log_errors=True, reraise_exceptions=UtilityModuleError)
    @log_execution(include_args=False, include_result=False)
    @validate_inputs(dependency='non_empty_string')
    def add_dependency(self, dependency: str) -> None:
        """Add a dependency to the module if not already present."""
        if not dependency or not dependency.strip():
            raise ValidationError("Dependency cannot be empty")
        if dependency in self.dependencies:
            raise DependencyError(f"Dependency '{dependency}' already exists in module")

        self.dependencies.append(dependency)
        self.updated_at = datetime.utcnow()

    @handle_errors(log_errors=True, reraise_exceptions=UtilityModuleError)
    @log_execution(include_args=False, include_result=False)
    @validate_inputs(dependency='non_empty_string')
    def remove_dependency(self, dependency: str) -> bool:
        """Remove a dependency from the module."""
        if not dependency or not dependency.strip():
            raise ValidationError("Dependency cannot be empty")
        if dependency not in self.dependencies:
            raise DependencyError(f"Dependency '{dependency}' not found in module")

        self.dependencies.remove(dependency)
        self.updated_at = datetime.utcnow()
        return True
