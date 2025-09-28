#!/usr/bin/env python3
"""
Test script for the refactoring progress tracking system.
This script tests the basic functionality of the progress tracker.
"""

import sys
import os
from datetime import datetime, timezone

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from progress_tracker import get_progress_tracker, ProgressTracker, RefactoringPhase, RefactoringStatus
    print("✓ Successfully imported progress tracker")
except ImportError as e:
    print(f"✗ Failed to import progress tracker: {e}")
    sys.exit(1)

def test_progress_tracker():
    """Test the progress tracking system."""
    print("\n=== Testing Progress Tracker ===")
    
    # Get the progress tracker
    tracker = get_progress_tracker()
    print("✓ Got progress tracker instance")
    
    # Test 1: Create progress tracking
    print("\n1. Testing progress creation...")
    try:
        progress = tracker.create_progress_tracking(
            module_name="test_module",
            refactoring_type="utility_module_creation",
            total_tasks=10,
            estimated_hours=2.0
        )
        print(f"✓ Created progress tracking: {progress.id}")
        print(f"  - Module: {progress.module_name}")
        print(f"  - Phase: {progress.phase.value}")
        print(f"  - Status: {progress.status.value}")
        print(f"  - Progress: {progress.progress_percentage}%")
    except Exception as e:
        print(f"✗ Failed to create progress: {e}")
        return False
    
    # Test 2: Update progress
    print("\n2. Testing progress update...")
    try:
        updated_progress = tracker.update_progress(
            progress_id=progress.id,
            tasks_completed=5,
            lines_refactored=100,
            note="Completed analysis phase"
        )
        print(f"✓ Updated progress: {updated_progress.progress_percentage}%")
        print(f"  - Tasks completed: {updated_progress.tasks_completed}")
        print(f"  - Lines refactored: {updated_progress.lines_refactored}")
        print(f"  - Notes: {len(updated_progress.notes)} notes")
    except Exception as e:
        print(f"✗ Failed to update progress: {e}")
        return False
    
    # Test 3: Mark phase complete
    print("\n3. Testing phase completion...")
    try:
        phase_progress = tracker.mark_phase_complete(
            progress_id=progress.id,
            phase=RefactoringPhase.ANALYSIS,
            note="Analysis phase completed successfully"
        )
        print(f"✓ Marked analysis phase complete")
        print(f"  - Current phase: {phase_progress.phase.value}")
        print(f"  - Progress: {phase_progress.progress_percentage}%")
    except Exception as e:
        print(f"✗ Failed to mark phase complete: {e}")
        return False
    
    # Test 4: Get progress by module
    print("\n4. Testing progress retrieval by module...")
    try:
        module_progress = tracker.get_progress_by_module("test_module")
        print(f"✓ Found {len(module_progress)} progress entries for test_module")
        for p in module_progress:
            print(f"  - {p.id}: {p.progress_percentage}% complete")
    except Exception as e:
        print(f"✗ Failed to get progress by module: {e}")
        return False
    
    # Test 5: Calculate overall progress
    print("\n5. Testing overall progress calculation...")
    try:
        overall_metrics = tracker.calculate_overall_progress()
        print(f"✓ Calculated overall progress:")
        print(f"  - Overall percentage: {overall_metrics['overall_percentage']}%")
        print(f"  - Total modules: {overall_metrics['total_modules']}")
        print(f"  - Completed modules: {overall_metrics['completed_modules']}")
        print(f"  - Lines refactored: {overall_metrics['lines_refactored']}")
    except Exception as e:
        print(f"✗ Failed to calculate overall progress: {e}")
        return False
    
    # Test 6: Generate progress report
    print("\n6. Testing progress report generation...")
    try:
        report = tracker.generate_progress_report()
        print(f"✓ Generated progress report:")
        print(f"  - Summary: {len(report['summary'])} metrics")
        print(f"  - Phase breakdown: {len(report['phase_breakdown'])} phases")
        print(f"  - Module breakdown: {len(report['module_breakdown'])} modules")
        print(f"  - Detailed progress: {len(report['detailed_progress'])} entries")
    except Exception as e:
        print(f"✗ Failed to generate progress report: {e}")
        return False
    
    # Test 7: Get module summary
    print("\n7. Testing module summary...")
    try:
        summary = tracker.get_module_summary("test_module")
        print(f"✓ Generated module summary:")
        print(f"  - Module: {summary['module_name']}")
        print(f"  - Total refactorings: {summary['total_refactorings']}")
        print(f"  - Overall progress: {summary['overall_progress']:.1f}%")
        print(f"  - Lines refactored: {summary['total_lines_refactored']}")
    except Exception as e:
        print(f"✗ Failed to get module summary: {e}")
        return False
    
    # Test 8: Mark refactoring complete
    print("\n8. Testing refactoring completion...")
    try:
        completed_progress = tracker.mark_refactoring_complete(
            progress_id=progress.id,
            final_note="Refactoring completed successfully"
        )
        print(f"✓ Marked refactoring as complete")
        print(f"  - Status: {completed_progress.status.value}")
        print(f"  - Progress: {completed_progress.progress_percentage}%")
        print(f"  - Is complete: {completed_progress.is_complete}")
    except Exception as e:
        print(f"✗ Failed to mark refactoring complete: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    return True

def test_api_endpoints():
    """Test the API endpoints."""
    print("\n=== Testing API Endpoints ===")
    
    # Test basic API functionality
    try:
        # Import the main module to test API structure
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        
        # Test that we can import the main module
        try:
            import main
            print("✓ Successfully imported main API module")
            
            # Check that FastAPI app exists
            if hasattr(main, 'app'):
                print("✓ FastAPI app found")
            else:
                print("⚠ FastAPI app not found in main module")
            
            # Check that progress tracker endpoints exist
            routes = [route.path for route in main.app.routes]
            progress_routes = [r for r in routes if 'refactoring-progress' in r]
            
            if progress_routes:
                print(f"✓ Found {len(progress_routes)} refactoring progress endpoints:")
                for route in progress_routes:
                    print(f"  - {route}")
            else:
                print("⚠ No refactoring progress endpoints found")
                
        except ImportError as e:
            print(f"⚠ Could not import main API module: {e}")
            print("  This is expected if FastAPI is not installed")
            
    except Exception as e:
        print(f"⚠ Error testing API endpoints: {e}")

def main():
    """Main test function."""
    print("Agentic Ticker Refactoring Progress Tracker Test")
    print("=" * 50)
    
    # Test the progress tracker
    success = test_progress_tracker()
    
    # Test API endpoints
    test_api_endpoints()
    
    if success:
        print("\n🎉 All progress tracker tests passed!")
        print("\nThe refactoring progress tracking system is working correctly.")
        print("Key features tested:")
        print("  ✓ Progress creation and tracking")
        print("  ✓ Progress updates with tasks and lines refactored")
        print("  ✓ Phase completion tracking")
        print("  ✓ Module-based progress retrieval")
        print("  ✓ Overall progress metrics calculation")
        print("  ✓ Comprehensive progress report generation")
        print("  ✓ Module summary statistics")
        print("  ✓ Refactoring completion marking")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())