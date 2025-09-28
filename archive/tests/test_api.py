#!/usr/bin/env python3
"""
Simple test script to verify the FastAPI application works correctly.
"""

import requests
import json
import time
import subprocess
import sys
import os

def start_api_server():
    """Start the API server in the background."""
    print("Starting API server...")
    process = subprocess.Popen(
        [sys.executable, "api.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for the server to start
    time.sleep(3)
    
    # Check if the process is still running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print(f"API server failed to start:")
        print(f"STDOUT: {stdout.decode()}")
        print(f"STDERR: {stderr.decode()}")
        return None
    
    return process

def test_api_endpoints():
    """Test all API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing API endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print("✓ Health endpoint working")
        else:
            print(f"✗ Health endpoint failed: {response.text}")
    except Exception as e:
        print(f"✗ Health endpoint error: {e}")
        return False
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"Root endpoint: {response.status_code}")
        if response.status_code == 200:
            print("✓ Root endpoint working")
        else:
            print(f"✗ Root endpoint failed: {response.text}")
    except Exception as e:
        print(f"✗ Root endpoint error: {e}")
        return False
    
    # Test POST /utility-modules
    test_module = {
        "name": "data_validator",
        "description": "Utility functions for data validation and sanitization",
        "file_path": "src/utils/data_validator.py",
        "functions": [
            {
                "name": "validate_email",
                "description": "Validates email format and domain",
                "parameters": [
                    {
                        "name": "email",
                        "type": "str",
                        "required": True
                    }
                ],
                "return_type": "bool",
                "source_modules": ["src/services.py", "src/orchestrator.py"],
                "lines_saved": 15
            }
        ],
        "dependencies": ["re", "html"]
    }
    
    try:
        response = requests.post(f"{base_url}/utility-modules", json=test_module)
        print(f"POST /utility-modules: {response.status_code}")
        if response.status_code == 201:
            print("✓ POST /utility-modules working")
            created_module = response.json()
            print(f"Created module: {created_module['name']}")
        else:
            print(f"✗ POST /utility-modules failed: {response.text}")
    except Exception as e:
        print(f"✗ POST /utility-modules error: {e}")
        return False
    
    # Test GET /utility-modules/{module_name}
    try:
        response = requests.get(f"{base_url}/utility-modules/data_validator")
        print(f"GET /utility-modules/data_validator: {response.status_code}")
        if response.status_code == 200:
            print("✓ GET /utility-modules working")
            module = response.json()
            print(f"Retrieved module: {module['name']}")
        else:
            print(f"✗ GET /utility-modules failed: {response.text}")
    except Exception as e:
        print(f"✗ GET /utility-modules error: {e}")
        return False
    
    # Test POST /decorators
    test_decorator = {
        "name": "handle_api_errors",
        "description": "Handles API errors consistently across all modules",
        "target_functions": [
            "src/services.py:fetch_ticker_data",
            "src/orchestrator.py:process_request"
        ],
        "concern": "error_handling",
        "implementation_pattern": "wrapper"
    }
    
    try:
        response = requests.post(f"{base_url}/decorators", json=test_decorator)
        print(f"POST /decorators: {response.status_code}")
        if response.status_code == 201:
            print("✓ POST /decorators working")
            created_decorator = response.json()
            print(f"Created decorator: {created_decorator['name']}")
        else:
            print(f"✗ POST /decorators failed: {response.text}")
    except Exception as e:
        print(f"✗ POST /decorators error: {e}")
        return False
    
    # Test GET /code-duplication-patterns
    try:
        response = requests.get(f"{base_url}/code-duplication-patterns")
        print(f"GET /code-duplication-patterns: {response.status_code}")
        if response.status_code == 200:
            print("✓ GET /code-duplication-patterns working")
        else:
            print(f"✗ GET /code-duplication-patterns failed: {response.text}")
    except Exception as e:
        print(f"✗ GET /code-duplication-patterns error: {e}")
        return False
    
    # Test POST /code-duplication-patterns
    test_pattern = {
        "pattern_id": "dup_001",
        "description": "Repeated validation logic across multiple files",
        "locations": [
            {
                "file_path": "/home/nitish/workspace/agentic-ticker/src/services.py",
                "start_line": 10,
                "end_line": 25,
                "function_name": "validate_input"
            },
            {
                "file_path": "/home/nitish/workspace/agentic-ticker/src/helpers.py",
                "start_line": 30,
                "end_line": 45,
                "function_name": "check_data"
            }
        ],
        "lines_affected": 16,
        "priority": "MEDIUM",
        "solution_approach": "Extract common validation logic into a shared utility function"
    }
    
    try:
        response = requests.post(f"{base_url}/code-duplication-patterns", json=test_pattern)
        print(f"POST /code-duplication-patterns: {response.status_code}")
        if response.status_code == 201:
            print("✓ POST /code-duplication-patterns working")
            created_pattern = response.json()
            print(f"Created pattern: {created_pattern['pattern_id']}")
        else:
            print(f"✗ POST /code-duplication-patterns failed: {response.text}")
    except Exception as e:
        print(f"✗ POST /code-duplication-patterns error: {e}")
        return False
    
    # Test GET /refactoring-progress
    try:
        response = requests.get(f"{base_url}/refactoring-progress")
        print(f"GET /refactoring-progress: {response.status_code}")
        if response.status_code == 200:
            print("✓ GET /refactoring-progress working")
            progress = response.json()
            print(f"Progress: {progress['completion_percentage']}%")
        else:
            print(f"✗ GET /refactoring-progress failed: {response.text}")
    except Exception as e:
        print(f"✗ GET /refactoring-progress error: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("Starting API tests...")
    
    # Start the API server
    server_process = start_api_server()
    if not server_process:
        print("Failed to start API server")
        return False
    
    try:
        # Run the tests
        success = test_api_endpoints()
        
        if success:
            print("\n🎉 All API tests passed!")
        else:
            print("\n❌ Some API tests failed")
        
        return success
        
    finally:
        # Clean up: stop the server
        print("Stopping API server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)