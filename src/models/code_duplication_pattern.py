from typing import List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator

from src.decorators import handle_errors, log_execution
from src.exceptions import (
    ValidationError, PatternDetectionError, AnalysisError,
    SimilarityError, CodeDuplicationError
)
from src.models.code_location import CodeLocation


class PatternType(str, Enum):
    """Types of code duplication patterns."""
    IDENTICAL_CODE = "identical_code"
    SIMILAR_STRUCTURE = "similar_structure"
    REPEATED_LOGIC = "repeated_logic"
    MAGIC_STRINGS = "magic_strings"
    HARDCODED_VALUES = "hardcoded_values"


class SeverityLevel(str, Enum):
    """Severity levels for code duplication patterns."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PatternStatus(str, Enum):
    """Status of code duplication pattern handling."""
    DETECTED = "detected"
    ANALYZED = "analyzed"
    REFACTORED = "refactored"
    IGNORED = "ignored"


class CodeDuplicationPattern(BaseModel):
    """Represents a code duplication pattern detected in the codebase."""

    id: str = Field(..., description="Unique identifier for the duplication pattern")
    pattern_type: PatternType = Field(
        ..., description="Type of code duplication pattern"
    )
    severity: SeverityLevel = Field(
        ..., description="Severity level of the duplication"
    )
    locations: List[CodeLocation] = Field(
        ...,
        description="List of locations where the duplication occurs (minimum 2)"
    )
    lines_affected: int = Field(
        ..., ge=1,
        description="Number of lines affected by the duplication"
    )
    suggested_refactoring: str = Field(
        ..., description="Suggested refactoring approach"
    )
    estimated_savings: int = Field(
        ..., ge=0,
        description="Estimated lines of code that could be saved"
    )
    detection_date: datetime = Field(
        default_factory=datetime.now,
        description="When the pattern was detected"
    )
    status: PatternStatus = Field(
        default=PatternStatus.DETECTED,
        description="Current status of the pattern"
    )

    @validator('locations')
    def validate_locations(cls, v):
        if len(v) < 2:
            raise PatternDetectionError(
                'At least 2 locations are required for a duplication pattern'
            )
        return v

    @validator('estimated_savings')
    def validate_savings(cls, v, values):
        if 'lines_affected' in values and v > values['lines_affected']:
            raise AnalysisError('estimated_savings cannot exceed lines_affected')
        return v

    @handle_errors(log_errors=True, reraise_exceptions=CodeDuplicationError)
    @log_execution(include_args=False, include_result=False)
    def add_location(self, location: CodeLocation) -> None:
        """Add a location where the duplication pattern occurs."""
        if not location:
            raise PatternDetectionError("Location cannot be None or empty")
        if any(loc.file_path == location.file_path and
               loc.start_line == location.start_line
               for loc in self.locations):
            raise PatternDetectionError("Location already exists in pattern")

        self.locations.append(location)

    @handle_errors(log_errors=True, reraise_exceptions=CodeDuplicationError)
    @log_execution(include_args=False, include_result=False)
    def calculate_similarity_score(self) -> float:
        """Calculate similarity score between all locations (0.0 to 1.0)."""
        if len(self.locations) < 2:
            raise SimilarityError(
                "At least 2 locations required for similarity calculation"
            )

        # Simple similarity calculation based on pattern type and lines affected
        base_score = 0.5
        if self.pattern_type == PatternType.IDENTICAL_CODE:
            base_score = 0.9
        elif self.pattern_type == PatternType.SIMILAR_STRUCTURE:
            base_score = 0.7
        elif self.pattern_type == PatternType.REPEATED_LOGIC:
            base_score = 0.6

        # Adjust based on number of locations (more locations = higher similarity)
        location_multiplier = min(len(self.locations) / 10.0, 1.0)

        return min(base_score * location_multiplier, 1.0)

    @handle_errors(log_errors=True, reraise_exceptions=CodeDuplicationError)
    @log_execution(include_args=False, include_result=False)
    def update_status(self, new_status: PatternStatus) -> None:
        """Update the status of the pattern."""
        if not isinstance(new_status, PatternStatus):
            raise ValidationError("Status must be a valid PatternStatus enum value")

        self.status = new_status

    @handle_errors(log_errors=True, reraise_exceptions=CodeDuplicationError)
    @log_execution(include_args=False, include_result=False)
    def get_refactoring_priority(self) -> int:
        """Calculate refactoring priority based on severity and estimated savings."""
        severity_weights = {
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4
        }

        base_priority = severity_weights.get(self.severity, 1)
        savings_bonus = min(self.estimated_savings // 10, 5)  # Bonus for high savings

        return base_priority + savings_bonus

    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "id": "dup_001",
                "pattern_type": "identical_code",
                "severity": "medium",
                "locations": [
                    {
                        "file_path": "src/utils.py",
                        "start_line": 10,
                        "end_line": 25,
                        "function_name": "validate_input",
                        "class_name": None,
                        "module_name": "utils"
                    },
                    {
                        "file_path": "src/helpers.py",
                        "start_line": 30,
                        "end_line": 45,
                        "function_name": "check_data",
                        "class_name": None,
                        "module_name": "helpers"
                    }
                ],
                "lines_affected": 16,
                "suggested_refactoring": "Extract common validation logic into a "
                                         "shared utility function",
                "estimated_savings": 12,
                "detection_date": "2025-09-27T10:30:00",
                "status": "detected"
            }
        }
