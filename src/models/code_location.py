from typing import Optional
from pydantic import BaseModel, Field, validator


class CodeLocation(BaseModel):
    """Represents a specific location in the codebase with detailed context."""

    file_path: str = Field(..., description="Path to the source file")
    start_line: int = Field(
        ..., ge=1, description="Starting line number (1-based indexing)"
    )
    end_line: int = Field(
        ..., ge=1, description="Ending line number (1-based indexing)"
    )
    function_name: Optional[str] = Field(
        None, description="Name of function containing this code"
    )
    class_name: Optional[str] = Field(
        None, description="Name of class containing this code"
    )
    module_name: str = Field(..., description="Name of the module")

    @validator("end_line")
    def validate_line_range(cls, v, values):
        """Validate that end_line is greater than or equal to start_line."""
        if "start_line" in values and v < values["start_line"]:
            raise ValueError("end_line must be greater than or equal to start_line")
        return v

    @validator("start_line", "end_line")
    def validate_positive_lines(cls, v):
        """Validate that line numbers are positive."""
        if v < 1:
            raise ValueError("Line numbers must be positive integers")
        return v

    @property
    def line_count(self) -> int:
        """Calculate the number of lines in this location."""
        return self.end_line - self.start_line + 1

    @property
    def has_function_context(self) -> bool:
        """Check if function context is available."""
        return self.function_name is not None

    @property
    def has_class_context(self) -> bool:
        """Check if class context is available."""
        return self.class_name is not None

    @property
    def full_context(self) -> str:
        """Get a string representation of the full context."""
        parts = []
        if self.module_name:
            parts.append(f"module: {self.module_name}")
        if self.class_name:
            parts.append(f"class: {self.class_name}")
        if self.function_name:
            parts.append(f"function: {self.function_name}")
        parts.append(f"lines: {self.start_line}-{self.end_line}")
        return ", ".join(parts)

    class Config:
        """Pydantic model configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "file_path": "src/utils/validation.py",
                "start_line": 15,
                "end_line": 28,
                "function_name": "validate_email",
                "class_name": "EmailValidator",
                "module_name": "validation",
            }
        }
