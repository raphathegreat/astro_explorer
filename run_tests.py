#!/usr/bin/env python3
"""
ISS Speed Analysis Dashboard - Easy Test Runner

Just run this file like any other Python script:
    python run_tests.py

This will run all tests and show you a nice summary.
"""

import sys
import os
import unittest
import subprocess
from pathlib import Path

def main():
    """Run all tests with a simple interface"""
    
    print("🚀 ISS Speed Analysis Dashboard - Test Suite")
    print("=" * 60)
    print("Running comprehensive tests to verify data integrity...")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('iss_speed_html_dashboard_v2_clean.py'):
        print("❌ Error: iss_speed_html_dashboard_v2_clean.py not found!")
        print("   Make sure you're in the correct directory.")
        return False
    
    # Check if tests directory exists
    if not os.path.exists('tests'):
        print("❌ Error: tests directory not found!")
        print("   The test suite hasn't been set up yet.")
        return False
    
    try:
        # Run the tests using unittest
        loader = unittest.TestLoader()
        start_dir = 'tests'
        suite = loader.discover(start_dir, pattern='test_*.py')
        
        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        passed = total_tests - failures - errors
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failures}")
        print(f"💥 Errors: {errors}")
        
        if total_tests > 0:
            success_rate = (passed / total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        # Show results
        if failures == 0 and errors == 0:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Data integrity verified from backend to frontend")
            print("✅ Your ISS Speed Analysis Dashboard is working correctly!")
            return True
        else:
            print(f"\n⚠️  {failures + errors} test(s) failed")
            print("Please review the output above for details.")
            
            if failures > 0:
                print("\n❌ FAILURES:")
                for test, traceback in result.failures:
                    print(f"  - {test}")
            
            if errors > 0:
                print("\n💥 ERRORS:")
                for test, traceback in result.errors:
                    print(f"  - {test}")
            
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the correct directory")
        print("2. Check that all test files exist")
        print("3. Try running: pip install -r requirements-test.txt")
        return False

def run_quick_test():
    """Run a quick test to verify the system is working"""
    print("🔍 Running quick system check...")
    
    try:
        # Try to import the main module
        import iss_speed_html_dashboard_v2_clean
        print("✅ Main module imports successfully")
        
        # Check if Flask app can be created
        app = iss_speed_html_dashboard_v2_clean.app
        print("✅ Flask app created successfully")
        
        # Check if test directory exists
        if os.path.exists('tests'):
            print("✅ Test directory found")
        else:
            print("❌ Test directory not found")
            return False
        
        print("✅ System check passed!")
        return True
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
        return False

if __name__ == "__main__":
    print("ISS Speed Analysis Dashboard - Test Runner")
    print("=" * 50)
    
    # First, run a quick system check
    if not run_quick_test():
        print("\n❌ System check failed. Please fix the issues above.")
        sys.exit(1)
    
    print()
    
    # Ask user what they want to do
    print("What would you like to do?")
    print("1. Run all tests (recommended)")
    print("2. Run quick tests only")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 Running full test suite...")
            success = main()
            sys.exit(0 if success else 1)
            
        elif choice == "2":
            print("\n🔍 Running quick tests...")
            # Run just a few key tests
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            
            # Add a few key tests
            try:
                from tests.unit.test_backend_functions import TestBackendFunctions
                suite.addTest(TestBackendFunctions('test_calculate_statistics'))
                suite.addTest(TestBackendFunctions('test_data_integrity_through_pipeline'))
                
                runner = unittest.TextTestRunner(verbosity=2)
                result = runner.run(suite)
                
                if result.wasSuccessful():
                    print("✅ Quick tests passed!")
                else:
                    print("❌ Quick tests failed!")
                    
            except Exception as e:
                print(f"❌ Error running quick tests: {e}")
                
        elif choice == "3":
            print("👋 Goodbye!")
            sys.exit(0)
            
        else:
            print("❌ Invalid choice. Please run the script again.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Test run cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
