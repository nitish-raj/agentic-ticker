"""
FastAPI application for the Agentic Ticker refactoring API.
Provides endpoints for utility module management, decorator creation, 
code duplication pattern detection, and refactoring progress tracking.
"""

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from enum import Enum
import re
import uuid
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the progress tracking system
try:
    from progress_tracker import get_progress_tracker, ProgressTracker
    # Import the enums from the main file to avoid conflicts
    progress_tracker_available = True
except ImportError:
    # Create a minimal progress tracker for testing
    class ProgressTracker:
        def generate_progress_report(self, **kwargs):
            return {
                "summary": {"error": "Progress tracker not available"},
                "phase_breakdown": {},
                "module_breakdown": {},
                "detailed_progress": []
            }
        def get_module_summary(self, module_name):
            return {
                "module_name": module_name,
                "total_refactorings": 0,
                "overall_progress": 0.0,
                "completed_refactorings": 0,
                "failed_refactorings": 0,
                "total_lines_refactored": 0,
                "average_completion_time": None,
                "current_refactorings": []
            }
        def create_progress_tracking(self, **kwargs):
            return type('obj', (object,), {
                'id': 'test', 
                'module_name': 'test', 
                'phase': type('enum', (), {'value': 'analysis'}), 
                'status': type('enum', (), {'value': 'in_progress'}), 
                'progress_percentage': 0, 
                'tasks_completed': 0, 
                'total_tasks': 0, 
                'started_at': datetime.now(timezone.utc), 
                'updated_at': datetime.now(timezone.utc)
            })()
        def update_progress(self, **kwargs):
            return type('obj', (object,), {
                'id': 'test', 
                'module_name': 'test', 
                'phase': type('enum', (), {'value': 'analysis'}), 
                'status': type('enum', (), {'value': 'in_progress'}), 
                'progress_percentage': 0, 
                'tasks_completed': 0, 
                'total_tasks': 0, 
                'lines_refactored': 0, 
                'updated_at': datetime.now(timezone.utc)
            })()
        def mark_refactoring_complete(self, progress_id, final_note):
            return type('obj', (object,), {
                'id': progress_id, 
                'module_name': 'test', 
                'status': type('enum', (), {'value': 'completed'}), 
                'progress_percentage': 100
            })()
        def mark_refactoring_failed(self, progress_id, failure_reason):
            return type('obj', (object,), {
                'id': progress_id, 
                'module_name': 'test', 
                'status': type('enum', (), {'value': 'failed'})
            })()
        def calculate_overall_progress(self):
            return {"overall_percentage": 0, "total_modules": 0}
    
    def get_progress_tracker():
        return ProgressTracker()
    
    progress_tracker_available = False

# Import the duplication detection system
try:
    from duplication_detector import scan_for_duplications, CodeDuplicationPattern as DetectionPattern
    from duplication_detector import PatternType, SeverityLevel, PatternStatus, CodeLocation
except ImportError:
    # Fallback if import fails
    scan_for_duplications = None

# Define enums and models inline to avoid import issues
class DecoratorConcern(str, Enum):
    ERROR_HANDLING = "error_handling"
    EVENT_REPORTING = "event_reporting"
    INPUT_VALIDATION = "input_validation"
    RETRY_LOGIC = "retry_logic"

class ImplementationPattern(str, Enum):
    WRAPPER = "wrapper"
    CONTEXT_MANAGER = "context_manager"
    CLASS_DECORATOR = "class_decorator"

class PatternType(str, Enum):
    IDENTICAL_CODE = "identical_code"
    SIMILAR_STRUCTURE = "similar_structure"
    REPEATED_LOGIC = "repeated_logic"
    MAGIC_STRINGS = "magic_strings"
    HARDCODED_VALUES = "hardcoded_values"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PatternStatus(str, Enum):
    DETECTED = "detected"
    ANALYZED = "analyzed"
    REFACTORED = "refactored"
    IGNORED = "ignored"

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

# Pydantic models
class FunctionParameter(BaseModel):
    name: str
    type: str
    default: Optional[Any] = None
    required: bool = True

class CodeLocation(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str] = None

class UtilityFunction(BaseModel):
    name: str = Field(..., pattern=r'^[a-z][a-z0-9_]*$')
    description: str
    parameters: List[FunctionParameter] = []
    return_type: str
    source_modules: List[str] = []
    lines_saved: int = Field(..., ge=1)

class UtilityModule(BaseModel):
    name: str = Field(..., pattern=r'^[a-z][a-z0-9_]*$')
    description: str
    file_path: str = Field(..., pattern=r'^src/[a-z][a-z0-9_/]*\.py$')
    functions: List[UtilityFunction] = []
    dependencies: List[str] = []

class Decorator(BaseModel):
    name: str = Field(..., pattern=r'^[a-z][a-z0-9_]*$')
    description: str
    target_functions: List[str] = []
    concern: DecoratorConcern
    implementation_pattern: ImplementationPattern

class CodeDuplicationPattern(BaseModel):
    id: str
    pattern_type: str = Field(..., pattern=r'^(identical_code|similar_structure|repeated_logic|magic_strings|hardcoded_values)$')
    severity: str = Field(..., pattern=r'^(low|medium|high|critical)$')
    locations: List[CodeLocation]
    lines_affected: int = Field(..., ge=1)
    suggested_refactoring: str
    estimated_savings: int = Field(..., ge=0)
    detection_date: Optional[datetime] = None
    status: str = Field(default="detected", pattern=r'^(detected|analyzed|refactored|ignored)$')
    description: Optional[str] = None  # For backward compatibility
    priority: Optional[str] = None  # For backward compatibility
    solution_approach: Optional[str] = None  # For backward compatibility
    line_numbers: Optional[List[int]] = None  # For backward compatibility

class RefactoringProgress(BaseModel):
    current_phase: str = Field(..., pattern=r'^(ANALYSIS_COMPLETE|UTILITIES_CREATED|DECORATORS_IMPLEMENTED|DUPLICATION_ELIMINATED|TESTS_UPDATED|VALIDATION_COMPLETE)$')
    completion_percentage: float = Field(..., ge=0, le=100)
    modules_created: int = Field(..., ge=0)
    decorators_implemented: int = Field(..., ge=0)
    duplication_eliminated: int = Field(..., ge=0)
    tests_updated: int = Field(..., ge=0)
    last_updated: Optional[datetime] = None

# Request/Response models for code duplication analysis
class CodeDuplicationAnalysisRequest(BaseModel):
    analysis_scope: str = Field(..., description="Directory path to scan (e.g., 'src/')")
    file_patterns: List[str] = Field(..., description="List of file patterns to include (e.g., ['*.py'])")
    min_duplication_lines: int = Field(default=5, ge=0, description="Minimum lines for duplication detection")
    ignore_comments: bool = Field(default=True, description="Whether to ignore comments in comparison")
    ignore_whitespace: bool = Field(default=True, description="Whether to ignore whitespace in comparison")
    include_tests: bool = Field(default=False, description="Whether to include test files")
    threshold_similarity: float = Field(default=0.8, ge=0.0, le=1.0, description="Threshold for similarity detection")

class CodeDuplicationAnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str
    analysis_scope: str
    file_patterns: List[str]
    min_duplication_lines: Optional[int] = None
    ignore_comments: Optional[bool] = None
    ignore_whitespace: Optional[bool] = None
    include_tests: Optional[bool] = None
    threshold_similarity: Optional[float] = None

# Create FastAPI app
app = FastAPI(
    title="Agentic Ticker Refactoring API",
    description="API for managing code refactoring operations including utility modules, decorators, and duplication patterns",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demonstration purposes
# In a real application, this would be replaced with a proper database
utility_modules: Dict[str, UtilityModule] = {}
decorators: Dict[str, Decorator] = {}
code_duplication_patterns: Dict[str, CodeDuplicationPattern] = {}
refactoring_progress: Dict[str, RefactoringProgress] = {}


# Utility Module Endpoints

@app.post("/utility-modules", response_model=UtilityModule, status_code=status.HTTP_201_CREATED)
async def create_utility_module(module: UtilityModule):
    """
    Create a new utility module.
    
    Args:
        module: UtilityModule data to create
        
    Returns:
        Created UtilityModule
        
    Raises:
        HTTPException: 400 for invalid data, 409 for duplicate name
    """
    try:
        # Validate module name uniqueness
        if module.name in utility_modules:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Module with name '{module.name}' already exists"
            )
        
        # Validate file path uniqueness
        existing_paths = {m.file_path for m in utility_modules.values()}
        if module.file_path in existing_paths:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Module with file path '{module.file_path}' already exists"
            )
        
        # Add to storage
        utility_modules[module.name] = module
        
        return module
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/utility-modules/{module_name}", response_model=UtilityModule)
async def get_utility_module(module_name: str):
    """
    Get a utility module by name.
    
    Args:
        module_name: Name of the utility module to retrieve
        
    Returns:
        UtilityModule data
        
    Raises:
        HTTPException: 404 if module not found
    """
    try:
        # Validate module name format
        if not module_name or not re.match(r'^[a-z][a-z0-9_]*$', module_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid module name format"
            )
        
        module = utility_modules.get(module_name)
        if not module:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Utility module '{module_name}' not found"
            )
        
        return module
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.put("/utility-modules/{module_name}", response_model=UtilityModule)
async def update_utility_module(module_name: str, module: UtilityModule):
    """
    Update an existing utility module.
    
    Args:
        module_name: Name of the utility module to update
        module: Updated UtilityModule data
        
    Returns:
        Updated UtilityModule
        
    Raises:
        HTTPException: 400 for invalid data, 404 if module not found
    """
    try:
        # Validate module name matches path parameter
        if module.name != module_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Module name in path must match module name in request body"
            )
        
        # Check if module exists
        if module_name not in utility_modules:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Utility module '{module_name}' not found"
            )
        
        # Update the module
        utility_modules[module_name] = module
        
        return module
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.delete("/utility-modules/{module_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_utility_module(module_name: str):
    """
    Delete a utility module.
    
    Args:
        module_name: Name of the utility module to delete
        
    Raises:
        HTTPException: 404 if module not found
    """
    try:
        # Validate module name format
        if not module_name or not re.match(r'^[a-z][a-z0-9_]*$', module_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid module name format"
            )
        
        if module_name not in utility_modules:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Utility module '{module_name}' not found"
            )
        
        # Delete the module
        del utility_modules[module_name]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Decorator Endpoints

@app.post("/decorators", response_model=Decorator, status_code=status.HTTP_201_CREATED)
async def create_decorator(decorator: Decorator):
    """
    Create a new decorator.
    
    Args:
        decorator: Decorator data to create
        
    Returns:
        Created Decorator
        
    Raises:
        HTTPException: 400 for invalid data, 409 for duplicate name
    """
    try:
        # Validate decorator name uniqueness
        if decorator.name in decorators:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Decorator with name '{decorator.name}' already exists"
            )
        
        # Add to storage
        decorators[decorator.name] = decorator
        
        return decorator
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Code Duplication Pattern Endpoints

@app.get("/code-duplication-patterns", response_model=List[CodeDuplicationPattern])
async def list_code_duplication_patterns(
    file_path: Optional[str] = None,
    pattern_type: Optional[str] = None,
    severity: Optional[str] = None
):
    """
    List code duplication patterns with optional filtering.
    
    Args:
        file_path: Optional filter by file path (substring match)
        pattern_type: Optional filter by pattern type
        severity: Optional filter by severity level
        
    Returns:
        List of CodeDuplicationPattern objects
    """
    try:
        patterns = list(code_duplication_patterns.values())
        
        # Apply filters
        if file_path:
            patterns = [
                p for p in patterns
                if any(file_path in loc.file_path for loc in p.locations)
            ]
        
        if pattern_type:
            if pattern_type not in ["identical_code", "similar_structure", "repeated_logic", "magic_strings", "hardcoded_values"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid pattern type"
                )
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        if severity:
            if severity not in ["low", "medium", "high", "critical"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid severity level"
                )
            patterns = [p for p in patterns if p.severity == severity]
        
        return patterns
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/code-duplication-patterns", response_model=CodeDuplicationAnalysisResponse, status_code=status.HTTP_202_ACCEPTED)
async def analyze_code_duplications(analysis_request: CodeDuplicationAnalysisRequest):
    """
    Analyze code for duplication patterns.
    
    Args:
        analysis_request: Code duplication analysis parameters
        
    Returns:
        Analysis job information with job_id for tracking
        
    Raises:
        HTTPException: 400 for invalid data, 500 if analysis fails
    """
    try:
        # Validate analysis scope exists
        if not os.path.exists(analysis_request.analysis_scope):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Analysis scope path does not exist: {analysis_request.analysis_scope}"
            )
        
        # Check if duplication detection is available
        if scan_for_duplications is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Code duplication detection system is not available"
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Perform the analysis (in a real system, this would be async)
        try:
            detected_patterns = scan_for_duplications(
                analysis_scope=analysis_request.analysis_scope,
                file_patterns=analysis_request.file_patterns,
                min_duplication_lines=analysis_request.min_duplication_lines,
                ignore_comments=analysis_request.ignore_comments,
                ignore_whitespace=analysis_request.ignore_whitespace,
                include_tests=analysis_request.include_tests,
                similarity_threshold=analysis_request.threshold_similarity
            )
            
            # Convert detected patterns to API format and store them
            for pattern in detected_patterns:
                api_pattern = CodeDuplicationPattern(
                    id=pattern.id,
                    pattern_type=pattern.pattern_type.value,
                    severity=pattern.severity.value,
                    locations=[
                        CodeLocation(
                            file_path=loc.file_path,
                            start_line=loc.start_line,
                            end_line=loc.end_line,
                            function_name=loc.function_name
                        ) for loc in pattern.locations
                    ],
                    lines_affected=pattern.lines_affected,
                    suggested_refactoring=pattern.suggested_refactoring,
                    estimated_savings=pattern.estimated_savings,
                    detection_date=pattern.detection_date,
                    status=pattern.status.value
                )
                code_duplication_patterns[pattern.id] = api_pattern
            
            status_msg = "completed"
            message = f"Analysis completed successfully. Found {len(detected_patterns)} duplication patterns."
            
        except Exception as analysis_error:
            status_msg = "failed"
            message = f"Analysis failed: {str(analysis_error)}"
        
        # Return response
        response_data = {
            "job_id": job_id,
            "status": status_msg,
            "message": message,
            "analysis_scope": analysis_request.analysis_scope,
            "file_patterns": analysis_request.file_patterns,
            "min_duplication_lines": analysis_request.min_duplication_lines,
            "ignore_comments": analysis_request.ignore_comments,
            "ignore_whitespace": analysis_request.ignore_whitespace,
            "include_tests": analysis_request.include_tests,
            "threshold_similarity": analysis_request.threshold_similarity
        }
        
        return CodeDuplicationAnalysisResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Refactoring Progress Endpoints

class ProgressReportResponse(BaseModel):
    """Response model for progress reports."""
    summary: Dict[str, Any]
    phase_breakdown: Dict[str, Any]
    module_breakdown: Dict[str, Any]
    detailed_progress: List[Dict[str, Any]]

class ModuleSummaryResponse(BaseModel):
    """Response model for module summaries."""
    module_name: str
    total_refactorings: int
    overall_progress: float
    completed_refactorings: int
    failed_refactorings: int
    total_lines_refactored: int
    average_completion_time: Optional[Dict[str, Any]] = None
    current_refactorings: List[Dict[str, Any]]

class CreateProgressRequest(BaseModel):
    """Request model for creating progress tracking."""
    module_name: str
    refactoring_type: str = Field(..., pattern=r'^(utility_module_creation|decorator_implementation|duplication_elimination)$')
    total_tasks: Optional[int] = None
    estimated_hours: Optional[float] = None

class UpdateProgressRequest(BaseModel):
    """Request model for updating progress."""
    tasks_completed: int = Field(..., ge=0)
    lines_refactored: Optional[int] = None
    phase: Optional[str] = Field(None, pattern=r'^(analysis|planning|implementation|verification|completion)$')
    status: Optional[str] = Field(None, pattern=r'^(not_started|in_progress|completed|failed|skipped)$')
    note: Optional[str] = None


@app.get("/refactoring-progress", response_model=Dict[str, Any])
async def get_refactoring_progress(
    module: Optional[str] = Query(None, description="Filter by module name"),
    start_date: Optional[datetime] = Query(None, description="Start date for time range filter (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="End date for time range filter (ISO format)"),
    include_completed: bool = Query(True, description="Include completed refactorings"),
    include_failed: bool = Query(True, description="Include failed refactorings")
):
    """
    Get comprehensive refactoring progress report with filtering options.
    
    Args:
        module: Filter by module name (partial match)
        start_date: Start date for time range filter
        end_date: End date for time range filter
        include_completed: Whether to include completed refactorings
        include_failed: Whether to include failed refactorings
    
    Returns:
        Comprehensive progress report with metrics, breakdowns, and detailed progress
    """
    try:
        tracker = get_progress_tracker()
        
        # Convert datetime strings to datetime objects if provided
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = start_date.replace(tzinfo=timezone.utc)
        if end_date:
            end_dt = end_date.replace(tzinfo=timezone.utc)
        
        # Generate comprehensive progress report
        report = tracker.generate_progress_report(
            module_filter=module,
            start_date=start_dt,
            end_date=end_dt,
            include_completed=include_completed,
            include_failed=include_failed
        )
        
        return report
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/refactoring-progress/modules/{module_name}", response_model=Dict[str, Any])
async def get_module_refactoring_summary(module_name: str):
    """
    Get refactoring summary for a specific module.
    
    Args:
        module_name: Name of the module to get summary for
    
    Returns:
        Module refactoring summary with statistics and current refactorings
    """
    try:
        tracker = get_progress_tracker()
        summary = tracker.get_module_summary(module_name)
        return summary
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/refactoring-progress", response_model=Dict[str, Any])
async def create_refactoring_progress(request: CreateProgressRequest):
    """
    Create new refactoring progress tracking.
    
    Args:
        request: Progress creation request
    
    Returns:
        Created progress tracking information
    """
    try:
        tracker = get_progress_tracker()
        
        # Create new progress tracking
        progress = tracker.create_progress_tracking(
            module_name=request.module_name,
            refactoring_type=request.refactoring_type,
            total_tasks=request.total_tasks,
            estimated_hours=request.estimated_hours
        )
        
        # Handle both real and fallback progress objects
        if hasattr(progress, 'id'):
            return {
                "id": progress.id,
                "module_name": progress.module_name,
                "phase": progress.phase.value if hasattr(progress.phase, 'value') else str(progress.phase),
                "status": progress.status.value if hasattr(progress.status, 'value') else str(progress.status),
                "progress_percentage": progress.progress_percentage,
                "tasks_completed": progress.tasks_completed,
                "total_tasks": progress.total_tasks,
                "started_at": progress.started_at.isoformat() if hasattr(progress.started_at, 'isoformat') else str(progress.started_at),
                "message": f"Created progress tracking for {request.module_name}"
            }
        else:
            # Fallback for testing
            return {
                "id": "test_id",
                "module_name": request.module_name,
                "phase": "analysis",
                "status": "in_progress",
                "progress_percentage": 0,
                "tasks_completed": 0,
                "total_tasks": request.total_tasks or 10,
                "started_at": datetime.utcnow().isoformat(),
                "message": f"Created progress tracking for {request.module_name} (test mode)"
            }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.put("/refactoring-progress/{progress_id}", response_model=Dict[str, Any])
async def update_refactoring_progress(progress_id: str, request: UpdateProgressRequest):
    """
    Update existing refactoring progress.
    
    Args:
        progress_id: ID of the progress to update
        request: Progress update request
    
    Returns:
        Updated progress information
    """
    try:
        tracker = get_progress_tracker()
        
        # Convert string phase/status to enums if provided
        phase_enum = None
        status_enum = None
        
        if request.phase:
            phase_enum = RefactoringPhase(request.phase)
        
        if request.status:
            status_enum = RefactoringStatus(request.status)
        
        # Update progress
        progress = tracker.update_progress(
            progress_id=progress_id,
            tasks_completed=request.tasks_completed,
            lines_refactored=request.lines_refactored,
            phase=phase_enum,
            status=status_enum,
            note=request.note
        )
        
        # Handle both real and fallback progress objects
        if hasattr(progress, 'id'):
            return {
                "id": progress.id,
                "module_name": progress.module_name,
                "phase": progress.phase.value if hasattr(progress.phase, 'value') else str(progress.phase),
                "status": progress.status.value if hasattr(progress.status, 'value') else str(progress.status),
                "progress_percentage": progress.progress_percentage,
                "tasks_completed": progress.tasks_completed,
                "total_tasks": progress.total_tasks,
                "lines_refactored": progress.lines_refactored,
                "updated_at": progress.updated_at.isoformat() if hasattr(progress.updated_at, 'isoformat') else str(progress.updated_at),
                "message": f"Updated progress for {progress.module_name}"
            }
        else:
            # Fallback for testing
            return {
                "id": progress_id,
                "module_name": "test_module",
                "phase": request.phase or "analysis",
                "status": request.status or "in_progress",
                "progress_percentage": (request.tasks_completed / 10) * 100 if request.tasks_completed else 0,
                "tasks_completed": request.tasks_completed,
                "total_tasks": 10,
                "lines_refactored": request.lines_refactored or 0,
                "updated_at": datetime.utcnow().isoformat(),
                "message": f"Updated progress for test_module (test mode)"
            }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/refactoring-progress/{progress_id}/complete", response_model=Dict[str, Any])
async def mark_refactoring_complete(progress_id: str, final_note: Optional[str] = None):
    """
    Mark refactoring as complete.
    
    Args:
        progress_id: ID of the progress to mark as complete
        final_note: Optional final note to add
    
    Returns:
        Completion confirmation
    """
    try:
        tracker = get_progress_tracker()
        progress = tracker.mark_refactoring_complete(progress_id, final_note)
        
        # Handle both real and fallback progress objects
        if hasattr(progress, 'id'):
            return {
                "id": progress.id,
                "module_name": progress.module_name,
                "status": progress.status.value if hasattr(progress.status, 'value') else str(progress.status),
                "progress_percentage": progress.progress_percentage,
                "message": f"Refactoring completed for {progress.module_name}"
            }
        else:
            # Fallback for testing
            return {
                "id": progress_id,
                "module_name": "test_module",
                "status": "completed",
                "progress_percentage": 100,
                "message": f"Refactoring marked as completed (test mode)"
            }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/refactoring-progress/{progress_id}/fail", response_model=Dict[str, Any])
async def mark_refactoring_failed(progress_id: str, failure_reason: str):
    """
    Mark refactoring as failed.
    
    Args:
        progress_id: ID of the progress to mark as failed
        failure_reason: Reason for failure
    
    Returns:
        Failure confirmation
    """
    try:
        tracker = get_progress_tracker()
        progress = tracker.mark_refactoring_failed(progress_id, failure_reason)
        
        # Handle both real and fallback progress objects
        if hasattr(progress, 'id'):
            return {
                "id": progress.id,
                "module_name": progress.module_name,
                "status": progress.status.value if hasattr(progress.status, 'value') else str(progress.status),
                "message": f"Refactoring marked as failed for {progress.module_name}"
            }
        else:
            # Fallback for testing
            return {
                "id": progress_id,
                "module_name": "test_module",
                "status": "failed",
                "message": f"Refactoring marked as failed: {failure_reason} (test mode)"
            }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/refactoring-progress/overall", response_model=Dict[str, Any])
async def get_overall_refactoring_progress():
    """
    Get overall refactoring progress metrics.
    
    Returns:
        Overall progress metrics and statistics
    """
    try:
        tracker = get_progress_tracker()
        metrics = tracker.calculate_overall_progress()
        
        return {
            "overall_metrics": metrics,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Health Check Endpoint

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "endpoints": {
            "utility_modules": len(utility_modules),
            "decorators": len(decorators),
            "code_duplication_patterns": len(code_duplication_patterns),
            "refactoring_progress": len(refactoring_progress)
        }
    }


# Root Endpoint

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        API information
    """
    return {
        "name": "Agentic Ticker Refactoring API",
        "version": "1.0.0",
        "description": "API for managing code refactoring operations",
        "documentation": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)