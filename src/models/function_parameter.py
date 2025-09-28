from typing import Any, Optional
from pydantic import BaseModel, Field


class FunctionParameter(BaseModel):
    """Represents a parameter in a utility function."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    default: Optional[Any] = Field(None, description="Default value for the parameter")
    required: bool = Field(True, description="Whether the parameter is required")

    class Config:
        use_enum_values = True
