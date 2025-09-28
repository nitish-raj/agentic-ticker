#!/usr/bin/env python3
"""
Standalone FastAPI application for the Agentic Ticker refactoring API.
Provides endpoints for utility module management, decorator creation, 
code duplication pattern detection, and refactoring progress tracking.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
import re

# Make uvicorn optional for import
uvicorn = None
try:
    import uvicorn
except ImportError:
    pass

# Define enums
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
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class PatternStatus(str, Enum):
    DETECTED = "detected"
    ANALYZED = "analyzed"
    REFACTORED = "refactored"
    IGNORED = "ignored"

class RefactoringPhase(str, Enum):
    ANALYSIS_COMPLETE = "ANALYSIS_COMPLETE"
    UTILITIES_CREATED = "UTILITIES_CREATED"
    DECORATORS_IMPLEMENTED = "DECORATORS_IMPLEMENTED"
    DUPLICATION_ELIMINATED = "DUPLICATION_ELIMINATED"
    TESTS_UPDATED = "TESTS_UPDATED"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"

# Pydantic models matching the OpenAPI contract
class FunctionParameter(BaseModel):
    name: str
    type: str
    default: Optional[Any] = None
    required: bool = True

    @validator('name')
    def validate_name(cls, v):
        if not v:
            raise ValueError("Parameter name cannot be empty")
        return v

class CodeLocation(BaseModel):
    file_path: str
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    function_name: Optional[str] = None

    @validator('end_line')
    def validate_end_line(cls, v, values):
        if 'start_line' in values and v < values['start_line']:
            raise ValueError("end_line must be greater than or equal to start_line")
        return v

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
    pattern_id: str
    description: str
    locations: List[CodeLocation]
    lines_affected: int = Field(..., ge=1)
    priority: SeverityLevel
    solution_approach: str

    @validator('locations')
    def validate_locations(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 locations are required for a duplication pattern")
        return v

class RefactoringProgress(BaseModel):
    current_phase: RefactoringPhase
    completion_percentage: float = Field(..., ge=0, le=100)
    modules_created: int = Field(..., ge=0)
    decorators_implemented: int = Field(..., ge=0)
    duplication_eliminated: int = Field(..., ge=0)
    tests_updated: int = Field(..., ge=0)
    last_updated: Optional[datetime] = None

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


def clear_storage():
    """Clear all in-memory storage. Used for testing."""
    utility_modules.clear()
    decorators.clear()
    code_duplication_patterns.clear()
    refactoring_progress.clear()


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
    # Validate module name uniqueness
    if module.name in utility_modules:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Module with name '{module.name}' already exists"
        )
    
    # Validate file path uniqueness
    existing_paths = {m.file_path for m in utility_modules.values()}
    if module.file_path in existing_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Module with file path '{module.file_path}' already exists"
        )
    
    # Add to storage
    utility_modules[module.name] = module
    
    return module


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


@app.delete("/utility-modules/{module_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_utility_module(module_name: str):
    """
    Delete a utility module.
    
    Args:
        module_name: Name of the utility module to delete
        
    Raises:
        HTTPException: 404 if module not found
    """
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
    # Validate decorator name uniqueness
    if decorator.name in decorators:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Decorator with name '{decorator.name}' already exists"
        )
    
    # Add to storage
    decorators[decorator.name] = decorator
    
    return decorator


# Code Duplication Pattern Endpoints

@app.get("/code-duplication-patterns", response_model=List[CodeDuplicationPattern])
async def list_code_duplication_patterns(priority: Optional[str] = None):
    """
    List code duplication patterns.
    
    Args:
        priority: Optional filter by priority level (HIGH, MEDIUM, LOW)
        
    Returns:
        List of CodeDuplicationPattern objects
    """
    patterns = list(code_duplication_patterns.values())
    
    # Filter by priority if provided
    if priority:
        if priority not in ["HIGH", "MEDIUM", "LOW"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid priority level. Must be HIGH, MEDIUM, or LOW"
            )
        patterns = [p for p in patterns if p.priority.value == priority]
    
    return patterns


@app.post("/code-duplication-patterns", response_model=CodeDuplicationPattern, status_code=status.HTTP_201_CREATED)
async def create_code_duplication_pattern(pattern: CodeDuplicationPattern):
    """
    Create a new code duplication pattern.
    
    Args:
        pattern: CodeDuplicationPattern data to create
        
    Returns:
        Created CodeDuplicationPattern
        
    Raises:
        HTTPException: 400 for invalid data
    """
    # Add to storage
    code_duplication_patterns[pattern.pattern_id] = pattern
    
    return pattern


# Refactoring Progress Endpoints

@app.get("/refactoring-progress", response_model=RefactoringProgress)
async def get_refactoring_progress():
    """
    Get current refactoring progress.
    
    Returns:
        Current RefactoringProgress data
    """
    # For demonstration, return a sample progress or create one if none exists
    if not refactoring_progress:
        # Create a sample refactoring progress
        sample_progress = RefactoringProgress(
            current_phase=RefactoringPhase.UTILITIES_CREATED,
            completion_percentage=65.0,
            modules_created=3,
            decorators_implemented=2,
            duplication_eliminated=5,
            tests_updated=8
        )
        refactoring_progress["sample"] = sample_progress
    
    # Return the first (and only) progress entry
    return list(refactoring_progress.values())[0]


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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)