#!/usr/bin/env python3
"""
E2E Test Runner for HRP Agent Workflows

Convenient script to run E2E tests with various configurations.
Provides shortcuts for common test execution scenarios.

Usage:
    python run_e2e.py                          # Run all E2E tests
    python run_e2e.py --pipeline               # Run pipeline tests only
    python run_e2e.py --fast                   # Run fast tests only
    python run_e2e.py --coverage               # Run with coverage report
    python run_e2e.py --watch                  # Watch mode for development
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ANSI color codes for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def run_command(cmd, description):
    """Run command and return success status."""
    print_info(f"Running: {description}")
    print(f"{Colors.WARNING}Command: {' '.join(cmd)}{Colors.ENDC}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print_success(f"{description} - PASSED")
        return True
    else:
        print_error(f"{description} - FAILED")
        return False


def run_all_tests(args):
    """Run all E2E tests."""
    cmd = ['pytest', 'tests/e2e/', '-v']
    if args.coverage:
        cmd.extend(['--cov=hrp.agents', '--cov-report=html', '--cov-report=term'])
    if args.marks:
        cmd.extend(['-m', args.marks])
    return run_command(cmd, "All E2E Tests")


def run_pipeline_tests(args):
    """Run event-driven pipeline tests."""
    cmd = [
        'pytest',
        'tests/e2e/test_agent_workflows_e2e.py::TestEventDrivenPipeline',
        '-v',
    ]
    if args.coverage:
        cmd.extend(['--cov=hrp.agents.pipeline_orchestrator', '--cov-report=term'])
    return run_command(cmd, "Pipeline Tests")


def run_concurrency_tests(args):
    """Run concurrency and resource management tests."""
    cmd = [
        'pytest',
        'tests/e2e/test_agent_workflows_e2e.py::TestConcurrency',
        '-v',
    ]
    return run_command(cmd, "Concurrency Tests")


def run_scheduled_tests(args):
    """Run scheduled workflow tests."""
    cmd = [
        'pytest',
        'tests/e2e/test_agent_workflows_e2e.py::TestScheduledWorkflows',
        '-v',
    ]
    return run_command(cmd, "Scheduled Workflow Tests")


def run_error_handling_tests(args):
    """Run error handling tests."""
    cmd = [
        'pytest',
        'tests/e2e/test_agent_workflows_e2e.py::TestErrorHandlingEdgeCases',
        '-v',
    ]
    return run_command(cmd, "Error Handling Tests")


def run_fast_tests(args):
    """Run fast tests only (skip slow/benchmark)."""
    cmd = [
        'pytest',
        'tests/e2e/',
        '-v',
        '-m',
        'not benchmark and not slow',
    ]
    return run_command(cmd, "Fast Tests Only")


def run_specific_test(args):
    """Run a specific test."""
    if not args.test:
        print_error("No test specified. Use --test TEST_NAME")
        return False

    cmd = ['pytest', args.test, '-v']
    if args.coverage:
        cmd.extend(['--cov=hrp.agents', '--cov-report=term'])
    return run_command(cmd, f"Specific Test: {args.test}")


def run_with_coverage(args):
    """Run tests with coverage report."""
    cmd = [
        'pytest',
        'tests/e2e/',
        '-v',
        '--cov=hrp.agents',
        '--cov-report=html',
        '--cov-report=term',
        '--cov-report=xml',
    ]
    if args.marks:
        cmd.extend(['-m', args.marks])

    success = run_command(cmd, "E2E Tests with Coverage")

    if success:
        print_info("\nCoverage reports generated:")
        print("  - HTML: htmlcov/index.html")
        print("  - XML: coverage.xml")
        print("\nOpen HTML report:")
        print(f"  {Colors.BOLD}open htmlcov/index.html{Colors.ENDC}")

    return success


def run_with_watch(args):
    """Run tests in watch mode (auto-rerun on file changes)."""
    print_info("Running tests in watch mode...")
    print_info("Press Ctrl+C to exit\n")

    cmd = ['ptw', 'tests/e2e/', '--runner', 'pytest', '-v']
    return run_command(cmd, "Watch Mode")


def run_diagnostics(args):
    """Run diagnostic checks."""
    print_header("E2E Test Diagnostics")

    # Check if pytest is installed
    try:
        import pytest
        version = pytest.__version__
        print_success(f"pytest installed (version {version})")
    except ImportError:
        print_error("pytest not installed")
        print_info("Install with: pip install pytest")
        return False

    # Check if HRP module is importable
    try:
        import hrp
        print_success("HRP module importable")
    except ImportError as e:
        print_error(f"HRP module not importable: {e}")
        print_info("Install HRP with: pip install -e .")
        return False

    # Check test directory exists
    test_dir = Path('tests/e2e')
    if test_dir.exists():
        print_success(f"Test directory exists: {test_dir}")
        test_files = list(test_dir.glob('test_*.py'))
        print_info(f"Found {len(test_files)} test files")
    else:
        print_error(f"Test directory not found: {test_dir}")
        return False

    # Check conftest.py
    conftest = test_dir / 'conftest.py'
    if conftest.exists():
        print_success("conftest.py found")
    else:
        print_warning("conftest.py not found (fixtures may not be available)")

    print()
    return True


def list_tests(args):
    """List all available tests."""
    cmd = ['pytest', 'tests/e2e/', '--collect-only', '-q']
    return run_command(cmd, "List Available Tests")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='E2E Test Runner for HRP Agent Workflows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Run all E2E tests
  %(prog)s --pipeline               Run pipeline tests only
  %(prog)s --fast                   Run fast tests only (skip slow/benchmark)
  %(prog)s --coverage               Run with coverage report
  %(prog)s --test test_name         Run specific test
  %(prog)s --watch                  Watch mode (auto-rerun on changes)
  %(prog)s --diagnostics            Run diagnostic checks
  %(prog)s --list                   List all available tests
        """,
    )

    # Action arguments
    parser.add_argument(
        '--pipeline',
        action='store_true',
        help='Run event-driven pipeline tests only'
    )
    parser.add_argument(
        '--concurrency',
        action='store_true',
        help='Run concurrency tests only'
    )
    parser.add_argument(
        '--scheduled',
        action='store_true',
        help='Run scheduled workflow tests only'
    )
    parser.add_argument(
        '--errors',
        action='store_true',
        help='Run error handling tests only'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run fast tests only (skip slow/benchmark)'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage report'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Run in watch mode (auto-rerun on file changes)'
    )
    parser.add_argument(
        '--diagnostics',
        action='store_true',
        help='Run diagnostic checks'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available tests'
    )

    # Options
    parser.add_argument(
        '--test',
        type=str,
        help='Run specific test (e.g., tests/e2e/test_agent_workflows_e2e.py::TestEventDrivenPipeline::test_complete_pipeline_flow)'
    )
    parser.add_argument(
        '-m',
        '--marks',
        type=str,
        help='Run tests with specific markers (e.g., "e2e", "not slow")'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Print header
    print_header("HRP Agent Workflows E2E Tests")

    # Run diagnostics if requested
    if args.diagnostics:
        return run_diagnostics(args)

    # List tests if requested
    if args.list:
        return list_tests(args)

    # Determine which tests to run
    success = True

    if args.test:
        success = run_specific_test(args)
    elif args.pipeline:
        success = run_pipeline_tests(args)
    elif args.concurrency:
        success = run_concurrency_tests(args)
    elif args.scheduled:
        success = run_scheduled_tests(args)
    elif args.errors:
        success = run_error_handling_tests(args)
    elif args.fast:
        success = run_fast_tests(args)
    elif args.watch:
        success = run_with_watch(args)
    elif args.coverage:
        success = run_with_coverage(args)
    else:
        # Default: run all tests
        success = run_all_tests(args)

    # Print summary
    print_header("Test Summary")
    if success:
        print_success("All tests passed!")
        return 0
    else:
        print_error("Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
