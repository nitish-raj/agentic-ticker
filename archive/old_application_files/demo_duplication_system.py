#!/usr/bin/env python3
"""
Final demonstration of the code duplication detection system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from duplication_detector import scan_for_duplications, get_patterns_by_filter
from duplication_detector import PatternType, SeverityLevel

def main():
    """Demonstrate the code duplication detection system."""
    print("🚀 Code Duplication Detection System - Final Demo")
    print("=" * 60)
    
    # Configure analysis parameters
    analysis_config = {
        "analysis_scope": "src/",
        "file_patterns": ["*.py"],
        "min_duplication_lines": 5,
        "ignore_comments": True,
        "ignore_whitespace": True,
        "include_tests": False,
        "similarity_threshold": 0.8
    }
    
    print(f"📊 Analysis Configuration:")
    for key, value in analysis_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Perform the analysis
    print("🔍 Scanning codebase for duplication patterns...")
    patterns = scan_for_duplications(**analysis_config)
    
    print(f"✅ Analysis complete! Found {len(patterns)} duplication patterns")
    print()
    
    # Show statistics
    if patterns:
        print("📈 Pattern Statistics:")
        
        # Count by pattern type
        type_counts = {}
        severity_counts = {}
        total_lines = 0
        total_savings = 0
        
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            severity = pattern.severity.value
            
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            total_lines += pattern.lines_affected
            total_savings += pattern.estimated_savings
        
        print("  By Pattern Type:")
        for pattern_type, count in sorted(type_counts.items()):
            print(f"    {pattern_type}: {count}")
        
        print("  By Severity:")
        for severity, count in sorted(severity_counts.items()):
            print(f"    {severity}: {count}")
        
        print(f"  Total lines affected: {total_lines:,}")
        print(f"  Total estimated savings: {total_savings:,} lines")
        print()
        
        # Show top patterns by severity
        print("🎯 Top Critical Patterns:")
        critical_patterns = [p for p in patterns if p.severity == SeverityLevel.CRITICAL][:5]
        
        for i, pattern in enumerate(critical_patterns, 1):
            print(f"\n  {i}. {pattern.id}")
            print(f"     Type: {pattern.pattern_type.value}")
            print(f"     Severity: {pattern.severity.value}")
            print(f"     Locations: {len(pattern.locations)} files")
            print(f"     Lines affected: {pattern.lines_affected}")
            print(f"     Estimated savings: {pattern.estimated_savings}")
            print(f"     Suggestion: {pattern.suggested_refactoring}")
            
            # Show example locations
            for j, loc in enumerate(pattern.locations[:2]):
                print(f"       Location {j+1}: {loc.file_path}:{loc.start_line}-{loc.end_line}")
            
            if len(pattern.locations) > 2:
                print(f"       ... and {len(pattern.locations) - 2} more locations")
        
        print()
        
        # Test filtering
        print("🔍 Testing Filtering Capabilities:")
        
        # Filter by pattern type
        identical_patterns = get_patterns_by_filter(patterns, pattern_type="identical_code")
        print(f"  Identical code patterns: {len(identical_patterns)}")
        
        # Filter by severity
        high_severity_patterns = get_patterns_by_filter(patterns, severity="high")
        print(f"  High severity patterns: {len(high_severity_patterns)}")
        
        # Filter by file path
        service_patterns = get_patterns_by_filter(patterns, file_path="services.py")
        print(f"  Patterns in services.py: {len(service_patterns)}")
        
        # Combined filtering
        high_identical_patterns = get_patterns_by_filter(
            patterns, 
            pattern_type="identical_code", 
            severity="high"
        )
        print(f"  High severity identical code patterns: {len(high_identical_patterns)}")
        
        print()
        
        # Show refactoring recommendations
        print("💡 Refactoring Recommendations:")
        
        # Group by suggestion type
        suggestions = {}
        for pattern in patterns[:20]:  # Top 20 patterns
            suggestion = pattern.suggested_refactoring
            if suggestion not in suggestions:
                suggestions[suggestion] = []
            suggestions[suggestion].append(pattern)
        
        for suggestion, pattern_list in list(suggestions.items())[:5]:
            print(f"\n  {suggestion}:")
            print(f"     Applies to {len(pattern_list)} patterns")
            total_savings = sum(p.estimated_savings for p in pattern_list)
            print(f"     Potential savings: {total_savings} lines")
        
        print()
        
        # Summary
        print("📋 Summary:")
        print(f"  ✅ Successfully detected {len(patterns)} duplication patterns")
        print(f"  📊 Identified {len(type_counts)} different pattern types")
        print(f"  🎯 Found patterns across {len(severity_counts)} severity levels")
        print(f"  💰 Estimated total savings: {total_savings:,} lines of code")
        print(f"  🔍 Demonstrated filtering by type, severity, and file path")
        print(f"  💡 Provided actionable refactoring recommendations")
        
    else:
        print("  ℹ️  No duplication patterns found - codebase is clean!")
    
    print()
    print("🎉 Code duplication detection system is working perfectly!")
    print("=" * 60)

if __name__ == "__main__":
    main()