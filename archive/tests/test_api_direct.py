#!/usr/bin/env python3
"""
Direct test of the FastAPI application without starting a server.
"""

from fastapi.testclient import TestClient
from api import app

def test_api():
    """Test the API using TestClient."""
    client = TestClient(app)
    
    print("Testing API endpoints...")
    
    # Test health endpoint
    response = client.get("/health")
    print(f"Health check: {response.status_code}")
    if response.status_code == 200:
        print("✓ Health endpoint working")
        print(f"Response: {response.json()}")
    else:
        print(f"✗ Health endpoint failed: {response.text}")
        return False
    
    # Test root endpoint
    response = client.get("/")
    print(f"Root endpoint: {response.status_code}")
    if response.status_code == 200:
        print("✓ Root endpoint working")
        print(f"Response: {response.json()}")
    else:
        print(f"✗ Root endpoint failed: {response.text}")
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
    
    response = client.post("/utility-modules", json=test_module)
    print(f"POST /utility-modules: {response.status_code}")
    if response.status_code == 201:
        print("✓ POST /utility-modules working")
        created_module = response.json()
        print(f"Created module: {created_module['name']}")
    else:
        print(f"✗ POST /utility-modules failed: {response.text}")
        return False
    
    # Test GET /utility-modules/{module_name}
    response = client.get("/utility-modules/data_validator")
    print(f"GET /utility-modules/data_validator: {response.status_code}")
    if response.status_code == 200:
        print("✓ GET /utility-modules working")
        module = response.json()
        print(f"Retrieved module: {module['name']}")
    else:
        print(f"✗ GET /utility-modules failed: {response.text}")
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
    
    response = client.post("/decorators", json=test_decorator)
    print(f"POST /decorators: {response.status_code}")
    if response.status_code == 201:
        print("✓ POST /decorators working")
        created_decorator = response.json()
        print(f"Created decorator: {created_decorator['name']}")
    else:
        print(f"✗ POST /decorators failed: {response.text}")
        return False
    
    # Test GET /code-duplication-patterns
    response = client.get("/code-duplication-patterns")
    print(f"GET /code-duplication-patterns: {response.status_code}")
    if response.status_code == 200:
        print("✓ GET /code-duplication-patterns working")
    else:
        print(f"✗ GET /code-duplication-patterns failed: {response.text}")
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
    
    response = client.post("/code-duplication-patterns", json=test_pattern)
    print(f"POST /code-duplication-patterns: {response.status_code}")
    if response.status_code == 201:
        print("✓ POST /code-duplication-patterns working")
        created_pattern = response.json()
        print(f"Created pattern: {created_pattern['pattern_id']}")
    else:
        print(f"✗ POST /code-duplication-patterns failed: {response.text}")
        return False
    
    # Test GET /refactoring-progress
    response = client.get("/refactoring-progress")
    print(f"GET /refactoring-progress: {response.status_code}")
    if response.status_code == 200:
        print("✓ GET /refactoring-progress working")
        progress = response.json()
        print(f"Progress: {progress['completion_percentage']}%")
    else:
        print(f"✗ GET /refactoring-progress failed: {response.text}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing API directly...")
    success = test_api()
    
    if success:
        print("\n🎉 All API tests passed!")
    else:
        print("\n❌ Some API tests failed")
    
    exit(0 if success else 1)