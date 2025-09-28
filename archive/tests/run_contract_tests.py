#!/usr/bin/env python3
"""
Run contract tests against the FastAPI application.
"""

import subprocess
import sys
import os

def run_test(test_file):
    """Run a specific test file."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    return result.returncode == 0

def main():
    """Run all contract tests."""
    test_files = [
        "tests/contract/test_utility_modules_post.py",
        "tests/contract/test_utility_modules_get.py",
        "tests/contract/test_utility_modules_put.py",
        "tests/contract/test_utility_modules_delete.py",
        "tests/contract/test_decorators_post.py",
        "tests/contract/test_code_duplication_patterns_get.py",
        "tests/contract/test_code_duplication_patterns_post.py",
        "tests/contract/test_refactoring_progress_get.py",
    ]
    
    results = {}
    
    for test_file in test_files:
        try:
            success = run_test(test_file)
            results[test_file] = success
            print(f"\n{'✓' if success else '✗'} {test_file}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"\n✗ {test_file}: ERROR - {e}")
            results[test_file] = False
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_file, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_file}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)