#!/usr/bin/env python3
"""
Integration test for the code duplication detection system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from duplication_detector import scan_for_duplications, get_patterns_by_filter
from duplication_detector import PatternType, SeverityLevel

def test_duplication_detection():
    """Test the core duplication detection functionality."""
    print("Testing code duplication detection system...")
    
    # Test basic detection
    patterns = scan_for_duplications(
        analysis_scope="src/",
        file_patterns=["*.py"],
        min_duplication_lines=5,
        include_tests=False
    )
    
    print(f"Found {len(patterns)} duplication patterns")
    
    # Test filtering
    print("\nTesting filtering by pattern type:")
    identical_patterns = [p for p in patterns if p.pattern_type == PatternType.IDENTICAL_CODE]
    print(f"Identical code patterns: {len(identical_patterns)}")
    
    print("\nTesting filtering by severity:")
    high_severity_patterns = [p for p in patterns if p.severity == SeverityLevel.HIGH]
    print(f"High severity patterns: {len(high_severity_patterns)}")
    
    print("\nTesting filtering by file path:")
    service_patterns = [p for p in patterns if any("services.py" in loc.file_path for loc in p.locations)]
    print(f"Patterns in services.py: {len(service_patterns)}")
    
    # Test get_patterns_by_filter function
    print("\nTesting get_patterns_by_filter function:")
    filtered_patterns = get_patterns_by_filter(
        patterns,
        pattern_type="identical_code",
        severity="high"
    )
    print(f"High severity identical code patterns: {len(filtered_patterns)}")
    
    # Show some example patterns
    if patterns:
        print("\nExample patterns:")
        for i, pattern in enumerate(patterns[:3]):
            print(f"\nPattern {i+1}:")
            print(f"  ID: {pattern.id}")
            print(f"  Type: {pattern.pattern_type.value}")
            print(f"  Severity: {pattern.severity.value}")
            print(f"  Locations: {len(pattern.locations)} files")
            print(f"  Lines affected: {pattern.lines_affected}")
            print(f"  Estimated savings: {pattern.estimated_savings}")
            print(f"  Suggestion: {pattern.suggested_refactoring}")
            
            # Show first location
            if pattern.locations:
                loc = pattern.locations[0]
                print(f"  Example location: {loc.file_path}:{loc.start_line}-{loc.end_line}")
    
    return len(patterns) > 0

def test_pattern_types():
    """Test that all pattern types are detected."""
    print("\nTesting pattern type detection...")
    
    patterns = scan_for_duplications(
        analysis_scope="src/",
        file_patterns=["*.py"],
        min_duplication_lines=3,
        include_tests=False
    )
    
    # Count patterns by type
    type_counts = {}
    for pattern in patterns:
        pattern_type = pattern.pattern_type.value
        type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
    
    print("Pattern type distribution:")
    for pattern_type, count in type_counts.items():
        print(f"  {pattern_type}: {count}")
    
    # Check that we have multiple types
    expected_types = ["identical_code", "similar_structure", "repeated_logic", "magic_strings", "hardcoded_values"]
    found_types = set(type_counts.keys())
    
    print(f"\nExpected types: {expected_types}")
    print(f"Found types: {list(found_types)}")
    
    return len(found_types) >= 3  # Should find at least 3 different types

if __name__ == "__main__":
    success1 = test_duplication_detection()
    success2 = test_pattern_types()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)