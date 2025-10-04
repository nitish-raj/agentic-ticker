from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import logging
from enum import Enum

logger = logging.getLogger(__name__)


# Define the enums and model classes directly
class RefactoringPhase(str, Enum):
    ANALYSIS = "analysis"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    COMPLETION = "completion"


class RefactoringStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RefactoringProgress:
    def __init__(self, **kwargs):
        # Set required attributes with defaults
        self.id = kwargs.get("id", "")
        self.module_name = kwargs.get("module_name", "")
        self.phase = kwargs.get("phase", RefactoringPhase.ANALYSIS)
        self.status = kwargs.get("status", RefactoringStatus.NOT_STARTED)
        self.progress_percentage = kwargs.get("progress_percentage", 0.0)
        self.tasks_completed = kwargs.get("tasks_completed", 0)
        self.total_tasks = kwargs.get("total_tasks", 0)
        self.lines_refactored = kwargs.get("lines_refactored", 0)
        self.estimated_completion = kwargs.get("estimated_completion", None)
        self.started_at = kwargs.get("started_at", datetime.now(timezone.utc))
        self.updated_at = kwargs.get("updated_at", datetime.now(timezone.utc))
        self.notes = kwargs.get("notes", [])

        # Set computed properties
        self.is_complete = self.status == RefactoringStatus.COMPLETED
        self.is_failed = self.status == RefactoringStatus.FAILED
        self.completion_ratio = (
            self.tasks_completed / self.total_tasks if self.total_tasks > 0 else 0.0
        )
        self.remaining_tasks = self.total_tasks - self.tasks_completed


class ProgressTracker:
    """Comprehensive refactoring progress tracking and reporting system."""

    def __init__(self):
        self._progress_store: Dict[str, RefactoringProgress] = {}
        self._phase_weights = {
            RefactoringPhase.ANALYSIS: 15,
            RefactoringPhase.PLANNING: 20,
            RefactoringPhase.IMPLEMENTATION: 40,
            RefactoringPhase.VERIFICATION: 20,
            RefactoringPhase.COMPLETION: 5,
        }
        self._task_definitions = {
            "utility_module_creation": {
                "analysis": [
                    "identify_duplication",
                    "analyze_dependencies",
                    "assess_complexity",
                ],
                "planning": [
                    "design_module_structure",
                    "define_interfaces",
                    "plan_migration",
                ],
                "implementation": [
                    "create_module",
                    "implement_functions",
                    "add_tests",
                    "update_imports",
                ],
                "verification": [
                    "run_tests",
                    "validate_functionality",
                    "check_performance",
                ],
                "completion": ["document_module", "update_readme", "mark_complete"],
            },
            "decorator_implementation": {
                "analysis": [
                    "identify_functions",
                    "analyze_error_patterns",
                    "assess_performance",
                ],
                "planning": [
                    "design_decorator_pattern",
                    "define_error_handling",
                    "plan_logging",
                ],
                "implementation": [
                    "create_decorators",
                    "implement_error_handling",
                    "add_logging",
                    "add_validation",
                ],
                "verification": [
                    "test_decorators",
                    "validate_error_handling",
                    "check_performance",
                ],
                "completion": [
                    "document_decorators",
                    "update_examples",
                    "mark_complete",
                ],
            },
            "duplication_elimination": {
                "analysis": ["scan_duplication", "analyze_patterns", "assess_impact"],
                "planning": [
                    "plan_refactoring",
                    "design_abstractions",
                    "prioritize_targets",
                ],
                "implementation": [
                    "extract_functions",
                    "create_utilities",
                    "consolidate_code",
                    "update_references",
                ],
                "verification": [
                    "validate_functionality",
                    "run_tests",
                    "check_coverage",
                ],
                "completion": [
                    "cleanup_old_code",
                    "update_documentation",
                    "mark_complete",
                ],
            },
        }

    def create_progress_tracking(
        self,
        module_name: str,
        refactoring_type: str,
        total_tasks: Optional[int] = None,
        estimated_hours: Optional[float] = None,
    ) -> RefactoringProgress:
        """Create new progress tracking for a refactoring operation."""
        if refactoring_type not in self._task_definitions:
            raise ValueError(f"Unknown refactoring type: {refactoring_type}")

        # Calculate total tasks if not provided
        if total_tasks is None:
            tasks_by_phase = self._task_definitions[refactoring_type]
            total_tasks = sum(len(tasks) for tasks in tasks_by_phase.values())

        # Generate unique ID
        progress_id = f"{refactoring_type}_{module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate estimated completion time
        estimated_completion = None
        if estimated_hours:
            estimated_completion = datetime.now(timezone.utc) + timedelta(
                hours=estimated_hours
            )

        progress = RefactoringProgress(
            id=progress_id,
            module_name=module_name,
            phase=RefactoringPhase.ANALYSIS,
            status=RefactoringStatus.NOT_STARTED,
            progress_percentage=0.0,
            tasks_completed=0,
            total_tasks=total_tasks,
            lines_refactored=0,
            estimated_completion=estimated_completion,
            started_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            notes=[f"Started {refactoring_type} refactoring for {module_name}"],
        )

        self._progress_store[progress_id] = progress
        logger.info(f"Created progress tracking: {progress_id}")
        return progress

    def update_progress(
        self,
        progress_id: str,
        tasks_completed: int,
        lines_refactored: Optional[int] = None,
        phase: Optional[RefactoringPhase] = None,
        status: Optional[RefactoringStatus] = None,
        note: Optional[str] = None,
    ) -> RefactoringProgress:
        """Update progress for a specific refactoring operation."""
        if progress_id not in self._progress_store:
            raise ValueError(f"Progress tracking not found: {progress_id}")

        progress = self._progress_store[progress_id]

        if tasks_completed > progress.total_tasks:
            raise ValueError("tasks_completed cannot exceed total_tasks")

        progress.tasks_completed = tasks_completed
        progress.progress_percentage = (
            (tasks_completed / progress.total_tasks) * 100
            if progress.total_tasks > 0
            else 0.0
        )

        if lines_refactored is not None:
            progress.lines_refactored = lines_refactored

        if phase is not None:
            progress.phase = phase

        if status is not None:
            progress.status = status

        if note is not None:
            progress.notes.append(note)

        progress.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Updated progress for {progress_id}: {tasks_completed}/{progress.total_tasks} tasks"
        )
        return progress

    def mark_phase_complete(
        self, progress_id: str, phase: RefactoringPhase, note: Optional[str] = None
    ) -> RefactoringProgress:
        """Mark a specific phase as complete and advance to next phase."""
        if progress_id not in self._progress_store:
            raise ValueError(f"Progress tracking not found: {progress_id}")

        progress = self._progress_store[progress_id]

        # Advance to next phase
        phases = list(RefactoringPhase)
        current_index = phases.index(phase)
        next_index = min(current_index + 1, len(phases) - 1)
        next_phase = phases[next_index]

        # Update progress based on phase completion
        self.update_progress(
            progress_id=progress_id,
            tasks_completed=progress.tasks_completed,
            phase=next_phase,
            status=(
                RefactoringStatus.IN_PROGRESS
                if next_phase != RefactoringPhase.COMPLETION
                else RefactoringStatus.COMPLETED
            ),
            note=note or f"Completed {phase.value} phase",
        )

        logger.info(f"Marked phase {phase.value} complete for {progress_id}")
        return progress

    def mark_refactoring_complete(
        self, progress_id: str, final_note: Optional[str] = None
    ) -> RefactoringProgress:
        """Mark refactoring as complete."""
        if progress_id not in self._progress_store:
            raise ValueError(f"Progress tracking not found: {progress_id}")

        progress = self._progress_store[progress_id]

        if progress.status == RefactoringStatus.COMPLETED:
            raise ValueError("Refactoring is already complete")

        progress.status = RefactoringStatus.COMPLETED
        progress.phase = RefactoringPhase.COMPLETION
        progress.progress_percentage = 100.0
        progress.tasks_completed = progress.total_tasks

        if final_note:
            progress.notes.append(final_note)

        progress.updated_at = datetime.now(timezone.utc)

        logger.info(f"Marked refactoring complete for {progress_id}")
        return progress

    def mark_refactoring_failed(
        self, progress_id: str, failure_reason: str
    ) -> RefactoringProgress:
        """Mark refactoring as failed."""
        if progress_id not in self._progress_store:
            raise ValueError(f"Progress tracking not found: {progress_id}")

        progress = self._progress_store[progress_id]

        if progress.status == RefactoringStatus.FAILED:
            raise ValueError("Refactoring is already marked as failed")

        progress.status = RefactoringStatus.FAILED
        progress.notes.append(f"Failed: {failure_reason}")
        progress.updated_at = datetime.now(timezone.utc)

        logger.error(f"Marked refactoring failed for {progress_id}: {failure_reason}")
        return progress

    def get_progress(self, progress_id: str) -> Optional[RefactoringProgress]:
        """Get progress by ID."""
        return self._progress_store.get(progress_id)

    def get_all_progress(self) -> List[RefactoringProgress]:
        """Get all progress tracking entries."""
        return list(self._progress_store.values())

    def get_progress_by_module(self, module_name: str) -> List[RefactoringProgress]:
        """Get all progress entries for a specific module."""
        return [
            p for p in self._progress_store.values() if p.module_name == module_name
        ]

    def get_progress_by_phase(
        self, phase: RefactoringPhase
    ) -> List[RefactoringProgress]:
        """Get all progress entries in a specific phase."""
        return [p for p in self._progress_store.values() if p.phase == phase]

    def get_progress_by_status(
        self, status: RefactoringStatus
    ) -> List[RefactoringProgress]:
        """Get all progress entries with a specific status."""
        return [p for p in self._progress_store.values() if p.status == status]

    def get_progress_by_time_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[RefactoringProgress]:
        """Get progress entries within a time range."""
        return [
            p
            for p in self._progress_store.values()
            if start_date <= p.started_at <= end_date
        ]

    def delete_progress(self, progress_id: str) -> bool:
        """Delete progress tracking entry."""
        if progress_id in self._progress_store:
            del self._progress_store[progress_id]
            logger.info(f"Deleted progress tracking: {progress_id}")
            return True
        return False

    def calculate_overall_progress(self) -> Dict[str, Any]:
        """Calculate overall refactoring progress across all modules."""
        all_progress = self.get_all_progress()

        if not all_progress:
            return {
                "overall_percentage": 0.0,
                "total_modules": 0,
                "completed_modules": 0,
                "in_progress_modules": 0,
                "failed_modules": 0,
                "total_tasks": 0,
                "completed_tasks": 0,
                "lines_refactored": 0,
            }

        total_modules = len(all_progress)
        completed_modules = len([p for p in all_progress if p.is_complete])
        failed_modules = len([p for p in all_progress if p.is_failed])
        in_progress_modules = total_modules - completed_modules - failed_modules

        total_tasks = sum(p.total_tasks for p in all_progress)
        completed_tasks = sum(p.tasks_completed for p in all_progress)
        lines_refactored = sum(p.lines_refactored for p in all_progress)

        # Calculate weighted overall percentage
        if total_tasks > 0:
            overall_percentage = (completed_tasks / total_tasks) * 100
        else:
            overall_percentage = (
                sum(p.progress_percentage for p in all_progress) / total_modules
            )

        return {
            "overall_percentage": round(overall_percentage, 2),
            "total_modules": total_modules,
            "completed_modules": completed_modules,
            "in_progress_modules": in_progress_modules,
            "failed_modules": failed_modules,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "lines_refactored": lines_refactored,
        }

    def generate_progress_report(
        self,
        module_filter: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_completed: bool = True,
        include_failed: bool = True,
    ) -> Dict[str, Any]:
        """Generate comprehensive progress report."""
        # Filter progress based on criteria
        filtered_progress = self.get_all_progress()

        if module_filter:
            filtered_progress = [
                p for p in filtered_progress if module_filter in p.module_name
            ]

        if start_date and end_date:
            filtered_progress = self.get_progress_by_time_range(start_date, end_date)

        if not include_completed:
            filtered_progress = [p for p in filtered_progress if not p.is_complete]

        if not include_failed:
            filtered_progress = [p for p in filtered_progress if not p.is_failed]

        # Calculate metrics
        overall_metrics = self.calculate_overall_progress()

        # Group by phase
        phase_breakdown = defaultdict(list)
        for progress in filtered_progress:
            phase_breakdown[progress.phase.value].append(
                {
                    "module_name": progress.module_name,
                    "progress_percentage": progress.progress_percentage,
                    "status": progress.status.value,
                    "tasks_completed": progress.tasks_completed,
                    "total_tasks": progress.total_tasks,
                    "started_at": progress.started_at.isoformat(),
                    "updated_at": progress.updated_at.isoformat(),
                }
            )

        # Group by module
        module_breakdown = defaultdict(list)
        for progress in filtered_progress:
            module_breakdown[progress.module_name].append(
                {
                    "phase": progress.phase.value,
                    "progress_percentage": progress.progress_percentage,
                    "status": progress.status.value,
                    "tasks_completed": progress.tasks_completed,
                    "total_tasks": progress.total_tasks,
                    "started_at": progress.started_at.isoformat(),
                    "updated_at": progress.updated_at.isoformat(),
                }
            )

        # Calculate trends
        recent_progress = [
            p
            for p in filtered_progress
            if (datetime.now(timezone.utc) - p.updated_at).days <= 7
        ]

        trend_analysis = {
            "recently_updated": len(recent_progress),
            "average_progress": (
                sum(p.progress_percentage for p in recent_progress)
                / len(recent_progress)
                if recent_progress
                else 0
            ),
            "completion_rate": (
                len([p for p in recent_progress if p.is_complete])
                / len(recent_progress)
                if recent_progress
                else 0
            ),
        }

        return {
            "summary": {
                "total_tracked_items": len(filtered_progress),
                "overall_progress": overall_metrics,
                "trend_analysis": trend_analysis,
                "report_generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "phase_breakdown": dict(phase_breakdown),
            "module_breakdown": dict(module_breakdown),
            "detailed_progress": [
                {
                    "id": p.id,
                    "module_name": p.module_name,
                    "phase": p.phase.value,
                    "status": p.status.value,
                    "progress_percentage": p.progress_percentage,
                    "tasks_completed": p.tasks_completed,
                    "total_tasks": p.total_tasks,
                    "lines_refactored": p.lines_refactored,
                    "estimated_completion": (
                        p.estimated_completion.isoformat()
                        if p.estimated_completion
                        else None
                    ),
                    "started_at": p.started_at.isoformat(),
                    "updated_at": p.updated_at.isoformat(),
                    "notes": p.notes,
                    "is_complete": p.is_complete,
                    "is_failed": p.is_failed,
                    "completion_ratio": p.completion_ratio,
                    "remaining_tasks": p.remaining_tasks,
                }
                for p in filtered_progress
            ],
        }

    def get_module_summary(self, module_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific module."""
        module_progress = self.get_progress_by_module(module_name)

        if not module_progress:
            return {
                "module_name": module_name,
                "total_refactorings": 0,
                "overall_progress": 0.0,
                "completed_refactorings": 0,
                "failed_refactorings": 0,
                "total_lines_refactored": 0,
                "average_completion_time": None,
                "current_refactorings": [],
            }

        completed = [p for p in module_progress if p.is_complete]
        failed = [p for p in module_progress if p.is_failed]

        # Calculate average completion time
        completion_times = []
        for progress in completed:
            if progress.estimated_completion:
                actual_duration = (
                    progress.updated_at - progress.started_at
                ).total_seconds() / 3600  # hours
                estimated_duration = (
                    progress.estimated_completion - progress.started_at
                ).total_seconds() / 3600
                completion_times.append(
                    {
                        "actual_hours": actual_duration,
                        "estimated_hours": estimated_duration,
                        "accuracy": (
                            abs(actual_duration - estimated_duration)
                            / estimated_duration
                            if estimated_duration > 0
                            else 0
                        ),
                    }
                )

        total_lines = sum(p.lines_refactored for p in module_progress)

        return {
            "module_name": module_name,
            "total_refactorings": len(module_progress),
            "overall_progress": sum(p.progress_percentage for p in module_progress)
            / len(module_progress),
            "completed_refactorings": len(completed),
            "failed_refactorings": len(failed),
            "total_lines_refactored": total_lines,
            "average_completion_time": (
                completion_times[0] if completion_times else None
            ),
            "current_refactorings": [
                {
                    "id": p.id,
                    "phase": p.phase.value,
                    "status": p.status.value,
                    "progress_percentage": p.progress_percentage,
                    "started_at": p.started_at.isoformat(),
                    "updated_at": p.updated_at.isoformat(),
                }
                for p in module_progress
                if not p.is_complete and not p.is_failed
            ],
        }


# Global progress tracker instance
_progress_tracker: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker()
    return _progress_tracker


def reset_progress_tracker() -> None:
    """Reset the global progress tracker instance."""
    global _progress_tracker
    _progress_tracker = None
