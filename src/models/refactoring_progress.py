from typing import List, Optional
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator

from src.decorators import handle_errors, log_execution, validate_inputs
from src.exceptions import (
    ValidationError, ProgressError, RefactoringStepError, CompletionError,
    RefactoringError
)


class RefactoringPhase(str, Enum):
    """Phases of the refactoring process."""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    COMPLETION = "completion"


class RefactoringStatus(str, Enum):
    """Status of the refactoring progress."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RefactoringProgress(BaseModel):
    """Represents the progress of a refactoring task for a specific module."""

    id: str = Field(..., description="Unique identifier for the refactoring progress")
    module_name: str = Field(..., description="Name of the module being refactored")
    phase: RefactoringPhase = Field(
        ..., description="Current phase of refactoring"
    )
    status: RefactoringStatus = Field(
        ..., description="Current status of the refactoring"
    )
    progress_percentage: float = Field(
        ..., ge=0.0, le=100.0,
        description="Progress percentage (0.0 to 100.0)"
    )
    tasks_completed: int = Field(..., ge=0, description="Number of tasks completed")
    total_tasks: int = Field(..., ge=0, description="Total number of tasks")
    lines_refactored: int = Field(
        ..., ge=0,
        description="Number of lines of code refactored"
    )
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time"
    )
    started_at: datetime = Field(..., description="When the refactoring started")
    updated_at: datetime = Field(..., description="When the progress was last updated")
    notes: List[str] = Field(
        default_factory=list,
        description="Progress notes and observations"
    )

    @validator('progress_percentage')
    def validate_progress_percentage(cls, v, values):
        """Validate progress percentage based on tasks."""
        if 'tasks_completed' in values and 'total_tasks' in values:
            tasks_completed = values['tasks_completed']
            total_tasks = values['total_tasks']
            if total_tasks > 0:
                expected_percentage = (tasks_completed / total_tasks) * 100
                if abs(v - expected_percentage) > 5.0:  # Allow 5% tolerance
                    raise ProgressError(
                        f"Progress percentage ({v}) doesn't match task "
                        f"completion ({tasks_completed}/{total_tasks} = "
                        f"{expected_percentage:.1f}%)"
                    )
        return v

    @validator('tasks_completed')
    def validate_tasks_completed(cls, v, values):
        """Validate tasks completed doesn't exceed total tasks."""
        if 'total_tasks' in values and v > values['total_tasks']:
            raise ProgressError("tasks_completed cannot exceed total_tasks")
        return v

    @validator('updated_at', always=True)
    def validate_updated_at(cls, v, values):
        """Ensure updated_at is always set to current time."""
        return datetime.utcnow()

    @validator('started_at')
    def validate_started_at(cls, v):
        """Validate started_at is not in the future."""
        now = datetime.now(timezone.utc)
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace('Z', '+00:00'))
        if v > now:
            raise ValidationError("started_at cannot be in the future")
        return v

    @root_validator(skip_on_failure=True)
    def validate_progress_consistency(cls, values):
        """Validate consistency between progress percentage and task completion."""
        tasks_completed = values.get('tasks_completed')
        total_tasks = values.get('total_tasks')
        progress_percentage = values.get('progress_percentage')

        if all([tasks_completed is not None, total_tasks is not None,
               progress_percentage is not None]):
            if total_tasks > 0:
                expected_pct = (tasks_completed / total_tasks) * 100
                if abs(progress_percentage - expected_pct) > 5.0:  # Allow 5% tolerance
                    msg = (f"Progress ({progress_percentage}%) vs tasks "
                           f"({tasks_completed}/{total_tasks} = {expected_pct:.1f}%)")
                    raise ProgressError(msg)

            if tasks_completed > total_tasks:
                raise ProgressError("tasks_completed cannot exceed total_tasks")

        return values

    @validator('estimated_completion')
    def validate_estimated_completion(cls, v, values):
        """Validate estimated_completion is after started_at."""
        if v is not None and 'started_at' in values and v <= values['started_at']:
            raise ValidationError("estimated_completion must be after started_at")
        return v

    @property
    def is_complete(self) -> bool:
        """Check if refactoring is complete."""
        return self.status == RefactoringStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if refactoring has failed."""
        return self.status == RefactoringStatus.FAILED

    @property
    def completion_ratio(self) -> float:
        """Calculate completion ratio as a decimal."""
        if self.total_tasks == 0:
            return 0.0
        return self.tasks_completed / self.total_tasks

    @property
    def remaining_tasks(self) -> int:
        """Calculate remaining tasks."""
        return self.total_tasks - self.tasks_completed

    @handle_errors(log_errors=True, reraise_exceptions=RefactoringError)
    @log_execution(include_args=False, include_result=False)
    @validate_inputs(note='non_empty_string')
    def add_note(self, note: str) -> None:
        """Add a progress note and update timestamp."""
        if not note or not note.strip():
            raise ValidationError("Note cannot be empty")
        self.notes.append(note)
        self.updated_at = datetime.utcnow()

    @handle_errors(log_errors=True, reraise_exceptions=RefactoringError)
    @log_execution(include_args=False, include_result=False)
    @validate_inputs(tasks_completed='positive_number')
    def update_progress(
        self,
        tasks_completed: int,
        lines_refactored: Optional[int] = None,
        phase: Optional[RefactoringPhase] = None,
        status: Optional[RefactoringStatus] = None,
        note: Optional[str] = None
    ) -> None:
        """Update progress with new values."""
        if tasks_completed > self.total_tasks:
            raise ProgressError("tasks_completed cannot exceed total_tasks")

        self.tasks_completed = tasks_completed
        self.progress_percentage = (
            (tasks_completed / self.total_tasks) * 100
            if self.total_tasks > 0 else 0.0
        )

        if lines_refactored is not None:
            if lines_refactored < 0:
                raise ValidationError("lines_refactored cannot be negative")
            self.lines_refactored = lines_refactored

        if phase is not None:
            self.phase = phase

        if status is not None:
            self.status = status

        if note is not None:
            self.add_note(note)

        self.updated_at = datetime.utcnow()

    @handle_errors(log_errors=True, reraise_exceptions=RefactoringError)
    @log_execution(include_args=False, include_result=False)
    def mark_complete(self, final_note: Optional[str] = None) -> None:
        """Mark refactoring as complete."""
        if self.status == RefactoringStatus.COMPLETED:
            raise CompletionError("Refactoring is already complete")

        self.status = RefactoringStatus.COMPLETED
        self.phase = RefactoringPhase.COMPLETION
        self.progress_percentage = 100.0
        self.tasks_completed = self.total_tasks

        if final_note:
            self.add_note(final_note)

        self.updated_at = datetime.utcnow()

    @handle_errors(log_errors=True, reraise_exceptions=RefactoringError)
    @log_execution(include_args=False, include_result=False)
    @validate_inputs(failure_reason='non_empty_string')
    def mark_failed(self, failure_reason: str) -> None:
        """Mark refactoring as failed."""
        if self.status == RefactoringStatus.FAILED:
            raise RefactoringStepError("Refactoring is already marked as failed")

        self.status = RefactoringStatus.FAILED
        self.add_note(f"Failed: {failure_reason}")
        self.updated_at = datetime.utcnow()

    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "id": "refactor_001",
                "module_name": "utils/validation",
                "phase": "implementation",
                "status": "in_progress",
                "progress_percentage": 65.0,
                "tasks_completed": 13,
                "total_tasks": 20,
                "lines_refactored": 250,
                "estimated_completion": "2025-09-28T15:00:00",
                "started_at": "2025-09-27T09:00:00",
                "updated_at": "2025-09-27T14:30:00",
                "notes": [
                    "Analysis phase completed",
                    "Planning phase completed",
                    "Implementation in progress"
                ]
            }
        }
