#!/usr/bin/env python3
"""
Verification script for MLflow feature version tracking.

This script verifies that the log_backtest function accepts and logs feature versions.
"""

import re
from pathlib import Path


def verify_implementation():
    """Verify the implementation by checking the source code."""
    mlflow_utils_path = Path("hrp/research/mlflow_utils.py")

    if not mlflow_utils_path.exists():
        print(f"❌ FAIL: {mlflow_utils_path} not found")
        return False

    content = mlflow_utils_path.read_text()

    checks = {
        "json import": r"import json",
        "feature_versions parameter": r"feature_versions:\s*dict\[str,\s*str\]\s*=\s*None",
        "feature_versions docstring": r"feature_versions:\s*Dict mapping feature names to versions",
        "log feature_versions JSON": r'mlflow\.log_param\("feature_versions",\s*json\.dumps\(feature_versions\)\)',
        "log individual versions": r'mlflow\.log_param\(f"feature_version_\{feature_name\}",\s*version\)',
        "check if feature_versions": r"if feature_versions:",
    }

    all_passed = True
    for check_name, pattern in checks.items():
        if re.search(pattern, content):
            print(f"✓ {check_name}")
        else:
            print(f"❌ FAIL: {check_name} not found")
            all_passed = False

    return all_passed


def main():
    """Run verification."""
    print("Verifying MLflow feature version tracking implementation...")
    print()

    if verify_implementation():
        print()
        print("✅ All code checks passed!")
        print()
        print("Implementation summary:")
        print("- Added feature_versions parameter to log_backtest()")
        print("- Logs feature versions as JSON string to 'feature_versions' param")
        print("- Logs individual feature versions to 'feature_version_<name>' params")
        print()
        print("Manual verification required:")
        print("1. Run a backtest with feature_versions={'momentum_20d': 'v1', 'volatility_60d': 'v1'}")
        print("2. Open MLflow UI: mlflow ui --backend-store-uri ~/hrp-data/mlflow/mlflow.db")
        print("3. Verify the run params include:")
        print("   - feature_versions: '{\"momentum_20d\": \"v1\", \"volatility_60d\": \"v1\"}'")
        print("   - feature_version_momentum_20d: 'v1'")
        print("   - feature_version_volatility_60d: 'v1'")
        return 0
    else:
        print()
        print("❌ Some checks failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
