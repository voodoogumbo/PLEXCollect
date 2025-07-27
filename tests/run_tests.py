#!/usr/bin/env python3
"""Test runner for PLEXCollect franchise system."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


def run_tests():
    """Run the complete test suite."""
    
    print("🧪 PLEXCollect Franchise System Test Suite")
    print("=" * 50)
    
    # Test configuration
    test_args = [
        str(Path(__file__).parent),  # Test directory
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        # "--cov=api",  # Coverage for api module (uncomment if coverage is installed)
        # "--cov=models",  # Coverage for models module
        # "--cov-report=html",  # HTML coverage report
    ]
    
    print(f"Running tests with args: {' '.join(test_args)}")
    print()
    
    # Run tests
    exit_code = pytest.main(test_args)
    
    print()
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return exit_code


def run_specific_test_category(category: str):
    """Run tests for a specific category."""
    
    test_files = {
        "franchise": "test_franchise_detection.py",
        "mega-batch": "test_mega_batch_optimization.py",
        "ui": "test_ui_integration.py",
        "integration": "test_collection_manager.py"
    }
    
    if category not in test_files:
        print(f"❌ Unknown test category: {category}")
        print(f"Available categories: {', '.join(test_files.keys())}")
        return 1
    
    test_file = Path(__file__).parent / test_files[category]
    
    print(f"🧪 Running {category} tests")
    print("=" * 30)
    
    exit_code = pytest.main([str(test_file), "-v"])
    
    print()
    if exit_code == 0:
        print(f"✅ {category} tests passed!")
    else:
        print(f"❌ {category} tests failed!")
    
    return exit_code


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PLEXCollect tests")
    parser.add_argument(
        "--category", 
        choices=["franchise", "mega-batch", "ui", "integration"],
        help="Run tests for a specific category"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (skip slow integration tests)"
    )
    
    args = parser.parse_args()
    
    if args.category:
        exit_code = run_specific_test_category(args.category)
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)