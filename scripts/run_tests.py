"""
Test Runner Script

Convenient script to run all tests with proper path setup and reporting.
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

def run_product_tests():
    """Run comprehensive product tests"""
    print("=" * 80)
    print("SUPPLY CHAIN INTELLIGENCE - TEST RUNNER")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        from tests.test_product import ProductTestSuite
        
        # Create and run test suite
        test_suite = ProductTestSuite(verbose=True)
        results = test_suite.run_all_tests()
        
        # Save results
        test_suite.save_results(results)
        
        # Return appropriate exit code
        return 0 if results['pass_rate'] >= 75 else 1
        
    except ImportError as e:
        print(f"Error importing test suite: {e}")
        print("Make sure you're running from the project root directory")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def run_quick_test():
    """Run quick functionality test"""
    print("Running quick functionality test...")
    
    try:
        from src import quick_analysis
        
        start_time = time.time()
        results = quick_analysis(verbose=False)
        duration = time.time() - start_time
        
        if results.get('pipeline_status') == 'success':
            print(f"Quick test PASSED ({duration:.1f}s)")
            print(f"Recommendation: {results.get('recommendation', {}).get('action', 'unknown')}")
            return True
        else:
            print(f"Quick test FAILED: {results.get('error_message', 'unknown')}")
            return False
            
    except Exception as e:
        print(f"Quick test ERROR: {e}")
        return False

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Supply Chain Intelligence Tests')
    parser.add_argument('--quick', action='store_true', help='Run quick functionality test only')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test suite')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    elif args.comprehensive or not args.quick:
        success = run_product_tests()
        sys.exit(0 if success else 1)
    
    else:
        print("No test specified. Use --quick or --comprehensive")
        sys.exit(1)

if __name__ == "__main__":
    main()
