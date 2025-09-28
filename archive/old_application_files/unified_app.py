#!/usr/bin/env python3
"""
Unified application that serves both Streamlit UI and FastAPI endpoints.
This allows the agentic ticker functionality to work alongside the refactoring API.
"""

import sys
import os
import threading
import time
import signal
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import FastAPI components
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError as e:
    print(f"Error importing FastAPI components: {e}")
    print("Please install fastapi and uvicorn: pip install fastapi uvicorn[standard]")
    sys.exit(1)

# Import Streamlit components
try:
    import streamlit as st
    from streamlit.web import cli as stcli
except ImportError as e:
    print(f"Error importing Streamlit: {e}")
    print("Please install streamlit: pip install streamlit")
    sys.exit(1)

# Import configuration system
try:
    from src.config import load_config, setup_logging, get_config
    from dotenv import load_dotenv, find_dotenv
except ImportError as e:
    print(f"Error importing configuration: {e}")
    print("Configuration system may not be available")
    load_config = None
    setup_logging = None
    get_config = None
    load_dotenv = None
    find_dotenv = None

# Global variables for server management
fastapi_server: Optional[uvicorn.Server] = None
streamlit_server = None
shutdown_event = threading.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    print("Starting unified application...")
    
    # Load configuration and setup logging
    load_dotenv(find_dotenv(), override=False)
    try:
        load_config()
        setup_logging()
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Warning: Configuration initialization failed: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down unified application...")
    shutdown_event.set()


# Create the unified FastAPI app
app = FastAPI(
    title="Agentic Ticker Unified API",
    description="Unified API serving both Streamlit UI and refactoring endpoints",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Import all the API endpoint functions from api.py and register them
# This ensures the contract tests can access the same endpoints

@app.post("/utility-modules", response_model=UtilityModule, status_code=status.HTTP_201_CREATED)
async def create_utility_module_endpoint(module: UtilityModule):
    """Create a new utility module"""
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
async def get_utility_module_endpoint(module_name: str):
    """Get a utility module by name"""
    import re
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
async def update_utility_module_endpoint(module_name: str, module: UtilityModule):
    """Update an existing utility module"""
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
async def delete_utility_module_endpoint(module_name: str):
    """Delete a utility module"""
    import re
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


@app.post("/decorators", response_model=Decorator, status_code=status.HTTP_201_CREATED)
async def create_decorator_endpoint(decorator: Decorator):
    """Create a new decorator"""
    # Validate decorator name uniqueness
    if decorator.name in decorators:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Decorator with name '{decorator.name}' already exists"
        )
    
    # Add to storage
    decorators[decorator.name] = decorator
    return decorator


@app.get("/code-duplication-patterns", response_model=list[CodeDuplicationPattern])
async def list_code_duplication_patterns_endpoint(priority: Optional[str] = None):
    """List code duplication patterns"""
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
async def create_code_duplication_pattern_endpoint(pattern: CodeDuplicationPattern):
    """Create a new code duplication pattern"""
    # Add to storage
    code_duplication_patterns[pattern.pattern_id] = pattern
    return pattern


@app.get("/refactoring-progress", response_model=RefactoringProgress)
async def get_refactoring_progress_endpoint():
    """Get current refactoring progress"""
    from datetime import datetime
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


@app.get("/health")
async def health_check_endpoint():
    """Health check endpoint"""
    from datetime import datetime
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


@app.get("/")
async def root_endpoint():
    """Root endpoint with API information"""
    return {
        "name": "Agentic Ticker Unified API",
        "version": "1.0.0",
        "description": "Unified API serving both Streamlit UI and refactoring endpoints",
        "documentation": "/docs",
        "health": "/health",
        "streamlit_ui": "http://localhost:8501"
    }


def run_streamlit():
    """Run Streamlit in a separate thread"""
    try:
        # Import the main Streamlit app
        sys.argv = ["streamlit", "run", "agentic_ticker.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
        stcli.main()
    except Exception as e:
        print(f"Streamlit error: {e}")


def run_fastapi():
    """Run FastAPI server"""
    global fastapi_server
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    fastapi_server = uvicorn.Server(config)
    
    try:
        fastapi_server.run()
    except Exception as e:
        print(f"FastAPI error: {e}")


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\nShutting down gracefully...")
    shutdown_event.set()
    
    if fastapi_server:
        fastapi_server.should_exit = True
    
    sys.exit(0)


def main():
    """Main function to run both servers"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Agentic Ticker Application")
    parser.add_argument("--api-only", action="store_true", help="Run only the FastAPI server")
    parser.add_argument("--ui-only", action="store_true", help="Run only the Streamlit UI")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=8501, help="Streamlit port (default: 8501)")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.ui_only:
        # Run only Streamlit
        print("Starting Streamlit UI only...")
        sys.argv = ["streamlit", "run", "agentic_ticker.py", "--server.port", str(args.ui_port), "--server.address", "0.0.0.0"]
        stcli.main()
    elif args.api_only:
        # Run only FastAPI
        print(f"Starting FastAPI server only on port {args.port}...")
        config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="info")
        server = uvicorn.Server(config)
        server.run()
    else:
        # Run both servers
        print("Starting unified application with both FastAPI and Streamlit...")
        print(f"FastAPI will be available at: http://localhost:{args.port}")
        print(f"Streamlit UI will be available at: http://localhost:{args.ui_port}")
        print(f"API documentation will be available at: http://localhost:{args.port}/docs")
        
        # Start FastAPI in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Give FastAPI time to start
        time.sleep(2)
        
        # Start Streamlit in the main thread (it needs to be main thread for some operations)
        try:
            sys.argv = ["streamlit", "run", "agentic_ticker.py", "--server.port", str(args.ui_port), "--server.address", "0.0.0.0"]
            stcli.main()
        except KeyboardInterrupt:
            print("\nShutting down...")
            shutdown_event.set()


if __name__ == "__main__":
    main()