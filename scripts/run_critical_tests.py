#!/usr/bin/env python3
"""
Critical Test Runner for MCP Memory System.

This script runs the critical MCP memory tests that validate the fixes for:
- MCP Retrieve Memory Tool response format validation
- Search Results response schema mismatch
- End-to-end memory operations
- Format validation edge cases

Usage:
    python scripts/run_critical_tests.py [--quick] [--verbose] [--test-type TYPE]
    
Options:
    --quick      Run only the fastest critical tests
    --verbose    Show detailed output
    --test-type  Run specific test type: format, critical, e2e, search, all
"""

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional


class CriticalTestRunner:
    """Runner for critical MCP memory system tests."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        
    def run_pytest(self, test_paths: List[str], markers: Optional[List[str]] = None, 
                   extra_args: List[str] = None) -> subprocess.CompletedProcess:
        """Run pytest with specified paths and markers."""
        cmd = [sys.executable, "-m", "pytest"]
        
        # Add test paths
        cmd.extend(test_paths)
        
        # Add markers
        if markers:
            marker_expr = " or ".join(markers)
            cmd.extend(["-m", marker_expr])
        
        # Add verbosity
        if self.verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        # Run from project root
        return subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=not self.verbose,
            text=True
        )
    
    def run_format_validation_tests(self) -> bool:
        """Run MCP format validation tests."""
        print("🧪 Running MCP format validation tests...")
        
        result = self.run_pytest([
            "tests/unit/test_mcp_format_validation.py"
        ])
        
        if result.returncode == 0:
            print("✅ Format validation tests passed")
            return True
        else:
            print("❌ Format validation tests failed")
            if not self.verbose:
                print("Output:", result.stdout)
                print("Errors:", result.stderr)
            return False
    
    def run_critical_feature_tests(self) -> bool:
        """Run critical MCP feature tests."""
        print("🧪 Running critical MCP feature tests...")
        
        result = self.run_pytest([
            "tests/integration/test_mcp_critical_features.py"
        ])
        
        if result.returncode == 0:
            print("✅ Critical feature tests passed")
            return True
        else:
            print("❌ Critical feature tests failed")
            if not self.verbose:
                print("Output:", result.stdout)
                print("Errors:", result.stderr)
            return False
    
    def run_e2e_retrieval_tests(self) -> bool:
        """Run end-to-end retrieval tests."""
        print("🧪 Running E2E retrieval tests...")
        
        result = self.run_pytest([
            "tests/integration/test_mcp_e2e_retrieval.py"
        ])
        
        if result.returncode == 0:
            print("✅ E2E retrieval tests passed")
            return True
        else:
            print("❌ E2E retrieval tests failed")
            if not self.verbose:
                print("Output:", result.stdout)
                print("Errors:", result.stderr)
            return False
    
    def run_search_functionality_tests(self) -> bool:
        """Run search functionality tests."""
        print("🧪 Running search functionality tests...")
        
        result = self.run_pytest([
            "tests/integration/test_mcp_search_functionality.py"
        ])
        
        if result.returncode == 0:
            print("✅ Search functionality tests passed")
            return True
        else:
            print("❌ Search functionality tests failed")
            if not self.verbose:
                print("Output:", result.stdout)
                print("Errors:", result.stderr)
            return False
    
    def run_quick_tests(self) -> bool:
        """Run only the quickest critical tests."""
        print("🏃 Running quick critical tests...")
        
        # Run just the format validation tests as they're fastest
        return self.run_format_validation_tests()
    
    def run_structured_thinking_tests(self) -> bool:
        """Run structured thinking tests."""
        print("🧪 Running structured thinking tests...")
        
        result = self.run_pytest([
            "tests/unit/test_structured_thinking.py"
        ])
        
        if result.returncode == 0:
            print("✅ Structured thinking tests passed")
            return True
        else:
            print("❌ Structured thinking tests failed")
            if not self.verbose:
                print("Output:", result.stdout)
                print("Errors:", result.stderr)
            return False
    
    def run_mcp_registry_tests(self) -> bool:
        """Run MCP registry/tool indexing tests."""
        print("🧪 Running MCP registry/tool indexing tests...")
        
        result = self.run_pytest([
            "tests/unit/test_mcp_tool_indexer.py"
        ])
        
        if result.returncode == 0:
            print("✅ MCP registry tests passed")
            return True
        else:
            print("❌ MCP registry tests failed")
            if not self.verbose:
                print("Output:", result.stdout)
                print("Errors:", result.stderr)
            return False
    
    def run_autocode_domain_tests(self) -> bool:
        """Run AutoCode domain tests."""
        print("🧪 Running AutoCode domain tests...")
        
        result = self.run_pytest([
            "tests/unit/test_autocode_domain.py"
        ])
        
        if result.returncode == 0:
            print("✅ AutoCode domain tests passed")
            return True
        else:
            print("❌ AutoCode domain tests failed")
            if not self.verbose:
                print("Output:", result.stdout)
                print("Errors:", result.stderr)
            return False
    
    def run_hook_system_tests(self) -> bool:
        """Run hook system tests."""
        print("🧪 Running hook system tests...")
        
        result = self.run_pytest([
            "tests/unit/test_hook_system.py"
        ])
        
        if result.returncode == 0:
            print("✅ Hook system tests passed")
            return True
        else:
            print("❌ Hook system tests failed")
            if not self.verbose:
                print("Output:", result.stdout)
                print("Errors:", result.stderr)
            return False
    
    def run_comprehensive_tests(self) -> bool:
        """Run comprehensive test suite for all major features."""
        print("🎯 Running comprehensive feature test suite...")
        
        start_time = time.perf_counter()
        results = []
        
        # Run all test suites
        results.append(("Format Validation", self.run_format_validation_tests()))
        results.append(("Critical Features", self.run_critical_feature_tests()))
        results.append(("E2E Retrieval", self.run_e2e_retrieval_tests()))
        results.append(("Search Functionality", self.run_search_functionality_tests()))
        results.append(("Structured Thinking", self.run_structured_thinking_tests()))
        results.append(("MCP Registry", self.run_mcp_registry_tests()))
        results.append(("AutoCode Domain", self.run_autocode_domain_tests()))
        results.append(("Hook System", self.run_hook_system_tests()))
        
        total_time = (time.perf_counter() - start_time)
        
        # Print summary
        print(f"\n📊 Comprehensive Test Summary (completed in {total_time:.1f}s):")
        print("=" * 60)
        
        passed_count = 0
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status:<12} {test_name}")
            if passed:
                passed_count += 1
        
        print("=" * 60)
        print(f"Result: {passed_count}/{len(results)} comprehensive test suites passed")
        
        return all(result[1] for result in results)
    
    def run_all_critical_tests(self) -> bool:
        """Run original critical MCP tests (memory system focus)."""
        print("🎯 Running critical MCP memory tests...")
        
        start_time = time.perf_counter()
        results = []
        
        # Run original critical test suites (memory-focused)
        results.append(("Format Validation", self.run_format_validation_tests()))
        results.append(("Critical Features", self.run_critical_feature_tests()))
        results.append(("E2E Retrieval", self.run_e2e_retrieval_tests()))
        results.append(("Search Functionality", self.run_search_functionality_tests()))
        
        total_time = (time.perf_counter() - start_time)
        
        # Print summary
        print(f"\n📊 Critical Test Summary (completed in {total_time:.1f}s):")
        print("=" * 50)
        
        passed_count = 0
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status:<10} {test_name}")
            if passed:
                passed_count += 1
        
        print("=" * 50)
        print(f"Result: {passed_count}/{len(results)} critical test suites passed")
        
        return all(result[1] for result in results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run critical MCP memory system tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the fastest critical tests"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Show detailed test output"
    )
    
    parser.add_argument(
        "--test-type", "-t",
        choices=["format", "critical", "e2e", "search", "structured_thinking", "mcp_registry", "autocode", "hooks", "comprehensive", "all"],
        default="all",
        help="Type of tests to run"
    )
    
    args = parser.parse_args()
    
    runner = CriticalTestRunner(verbose=args.verbose)
    
    print("🚀 MCP Memory System Critical Test Runner")
    print("=" * 50)
    
    success = False
    
    if args.quick:
        success = runner.run_quick_tests()
    elif args.test_type == "format":
        success = runner.run_format_validation_tests()
    elif args.test_type == "critical":
        success = runner.run_critical_feature_tests()
    elif args.test_type == "e2e":
        success = runner.run_e2e_retrieval_tests()
    elif args.test_type == "search":
        success = runner.run_search_functionality_tests()
    elif args.test_type == "structured_thinking":
        success = runner.run_structured_thinking_tests()
    elif args.test_type == "mcp_registry":
        success = runner.run_mcp_registry_tests()
    elif args.test_type == "autocode":
        success = runner.run_autocode_domain_tests()
    elif args.test_type == "hooks":
        success = runner.run_hook_system_tests()
    elif args.test_type == "comprehensive":
        success = runner.run_comprehensive_tests()
    else:  # all (original critical tests)
        success = runner.run_all_critical_tests()
    
    if success:
        print("\n🎉 All specified tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        print("\n🔧 Next steps:")
        print("  - Review the test output above for specific failures")
        print("  - Check if the MCP memory system fixes are still in place")
        print("  - Run with --verbose for more detailed output")
        print("  - Run individual test types with --test-type")
        sys.exit(1)


if __name__ == "__main__":
    main()