#!/usr/bin/env python3
"""
Test script for the code duplication detection API endpoints.
"""

import requests
import json
import sys

def test_api_endpoints():
    """Test the code duplication pattern API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing Code Duplication Detection API...")
    
    # Test POST endpoint - analyze code duplications
    print("\n1. Testing POST /code-duplication-patterns")
    post_data = {
        "analysis_scope": "src/",
        "file_patterns": ["*.py"],
        "min_duplication_lines": 5,
        "ignore_comments": True,
        "ignore_whitespace": True,
        "include_tests": False,
        "threshold_similarity": 0.8
    }
    
    try:
        response = requests.post(f"{base_url}/code-duplication-patterns", json=post_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 202:
            result = response.json()
            print(f"Job ID: {result.get('job_id')}")
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message')}")
            print(f"Patterns found: {result.get('message', '').split()[-2] if 'Found' in result.get('message', '') else 'unknown'}")
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server. Make sure it's running on port 8000.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    # Test GET endpoint - list patterns
    print("\n2. Testing GET /code-duplication-patterns")
    try:
        response = requests.get(f"{base_url}/code-duplication-patterns")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            patterns = response.json()
            print(f"Total patterns: {len(patterns)}")
            
            if patterns:
                # Show first pattern details
                first_pattern = patterns[0]
                print(f"First pattern ID: {first_pattern.get('id')}")
                print(f"Type: {first_pattern.get('pattern_type')}")
                print(f"Severity: {first_pattern.get('severity')}")
                print(f"Locations: {len(first_pattern.get('locations', []))}")
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server. Make sure it's running on port 8000.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    # Test GET with filters
    print("\n3. Testing GET /code-duplication-patterns with filters")
    filters = {
        "pattern_type": "identical_code",
        "severity": "high"
    }
    
    try:
        response = requests.get(f"{base_url}/code-duplication-patterns", params=filters)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            filtered_patterns = response.json()
            print(f"Filtered patterns: {len(filtered_patterns)}")
            
            # Verify filtering worked
            if filtered_patterns:
                all_match = all(
                    p.get('pattern_type') == 'identical_code' and 
                    p.get('severity') == 'high' 
                    for p in filtered_patterns
                )
                print(f"All patterns match filters: {all_match}")
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server. Make sure it's running on port 8000.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_api_endpoints()
    sys.exit(0 if success else 1)