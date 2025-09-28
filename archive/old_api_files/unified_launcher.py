#!/usr/bin/env python3
"""
Main entry point for the unified Agentic Ticker application.
This script can launch either the Streamlit UI, FastAPI server, or both.
"""

import sys
import os
import subprocess
import argparse
import signal
import time
import threading
from pathlib import Path

# Add the project directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_streamlit(port=8501):
    """Run the Streamlit application"""
    print(f"Starting Streamlit UI on port {port}...")
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "agentic_ticker.py", 
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit failed to start: {e}")
        return False
    except FileNotFoundError:
        print("Streamlit not found. Please install: pip install streamlit")
        return False
    return True


def run_fastapi(port=8000):
    """Run the FastAPI server"""
    print(f"Starting FastAPI server on port {port}...")
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "api:app", 
        "--host", "0.0.0.0", 
        "--port", str(port),
        "--reload"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FastAPI failed to start: {e}")
        return False
    except FileNotFoundError:
        print("uvicorn not found. Please install: pip install uvicorn[standard]")
        return False
    return True


def run_both(streamlit_port=8501, fastapi_port=8000):
    """Run both Streamlit and FastAPI in parallel"""
    print("Starting unified application with both Streamlit and FastAPI...")
    print(f"FastAPI will be available at: http://localhost:{fastapi_port}")
    print(f"Streamlit UI will be available at: http://localhost:{streamlit_port}")
    print(f"API documentation will be available at: http://localhost:{fastapi_port}/docs")
    
    # Create threads for both services
    fastapi_thread = threading.Thread(
        target=run_fastapi, 
        args=(fastapi_port,),
        daemon=True
    )
    
    # Start FastAPI in background thread
    fastapi_thread.start()
    
    # Give FastAPI time to start
    time.sleep(3)
    
    # Run Streamlit in main thread (it needs to be main thread for some operations)
    try:
        run_streamlit(streamlit_port)
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    # Wait for FastAPI thread to finish
    fastapi_thread.join(timeout=5)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Agentic Ticker - Unified Application Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both Streamlit UI and FastAPI API
  python unified_launcher.py
  
  # Run only FastAPI server
  python unified_launcher.py --api-only
  
  # Run only Streamlit UI
  python unified_launcher.py --ui-only
  
  # Custom ports
  python unified_launcher.py --api-port 8080 --ui-port 8502
        """
    )
    
    parser.add_argument(
        "--api-only", 
        action="store_true", 
        help="Run only the FastAPI server"
    )
    
    parser.add_argument(
        "--ui-only", 
        action="store_true", 
        help="Run only the Streamlit UI"
    )
    
    parser.add_argument(
        "--api-port", 
        type=int, 
        default=8000, 
        help="FastAPI port (default: 8000)"
    )
    
    parser.add_argument(
        "--ui-port", 
        type=int, 
        default=8501, 
        help="Streamlit port (default: 8501)"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run contract tests after starting servers"
    )
    
    args = parser.parse_args()
    
    # Validate port arguments
    if args.api_port < 1024 or args.api_port > 65535:
        print("Error: API port must be between 1024 and 65535")
        sys.exit(1)
    
    if args.ui_port < 1024 or args.ui_port > 65535:
        print("Error: UI port must be between 1024 and 65535")
        sys.exit(1)
    
    if args.api_port == args.ui_port:
        print("Error: API and UI ports must be different")
        sys.exit(1)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.ui_only:
            # Run only Streamlit
            success = run_streamlit(args.ui_port)
        elif args.api_only:
            # Run only FastAPI
            success = run_fastapi(args.api_port)
        else:
            # Run both
            run_both(args.ui_port, args.api_port)
            success = True
        
        # Run contract tests if requested
        if success and args.test:
            print("\nRunning contract tests...")
            test_cmd = [sys.executable, "run_contract_tests.py"]
            try:
                subprocess.run(test_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Contract tests failed: {e}")
                return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())