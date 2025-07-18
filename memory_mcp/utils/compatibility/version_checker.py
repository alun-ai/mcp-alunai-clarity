"""
Version compatibility checker for memory_mcp.
"""

import importlib
import importlib.metadata
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class CompatibilityReport:
    """Compatibility report for dependency versions."""
    
    compatible: bool
    issues: List[str]
    python_version: str


def check_python_version() -> Tuple[bool, Optional[str]]:
    """
    Check if the current Python version is compatible.
    
    Returns:
        Tuple of (is_compatible, error_message)
    """
    python_version = sys.version_info
    
    # We support Python 3.8 to 3.12
    if python_version.major != 3 or python_version.minor < 8 or python_version.minor > 12:
        return False, f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} is not supported. Please use Python 3.8-3.12."
    
    return True, None


def check_dependency_version(package_name: str, min_version: str, max_version: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a dependency version is within the expected range.
    
    Args:
        package_name: Name of the package to check
        min_version: Minimum supported version (inclusive)
        max_version: Maximum supported version (exclusive)
        
    Returns:
        Tuple of (is_compatible, error_message)
    """
    try:
        version = importlib.metadata.version(package_name)
        
        # Simple version comparison (assumes semantic versioning)
        min_parts = [int(x) for x in min_version.split('.')]
        max_parts = [int(x) for x in max_version.split('.')]
        version_parts = [int(x) for x in version.split('.')]
        
        # Check minimum version
        for i in range(len(min_parts)):
            if i >= len(version_parts):
                break
            if version_parts[i] < min_parts[i]:
                return False, f"{package_name} version {version} is lower than the minimum supported version {min_version}"
            if version_parts[i] > min_parts[i]:
                break
        
        # Check maximum version
        for i in range(len(max_parts)):
            if i >= len(version_parts):
                break
            if version_parts[i] >= max_parts[i]:
                return False, f"{package_name} version {version} is higher than the maximum supported version {max_version}"
            if version_parts[i] < max_parts[i]:
                break
        
        return True, None
    except importlib.metadata.PackageNotFoundError:
        return False, f"{package_name} is not installed"
    except Exception as e:
        return False, f"Error checking {package_name} version: {str(e)}"


def check_compatibility() -> CompatibilityReport:
    """
    Check compatibility of the current environment.
    
    Returns:
        CompatibilityReport with details about compatibility
    """
    issues = []
    
    # Check Python version
    python_compatible, python_error = check_python_version()
    if not python_compatible:
        issues.append(python_error)
    
    # Critical dependencies and their version ranges
    dependencies = {
        "numpy": ("1.20.0", "2.0.0"),
        "pydantic": ("2.4.0", "3.0.0"),
        "sentence-transformers": ("2.2.2", "3.0.0"),
        "hnswlib": ("0.7.0", "0.8.0"),
        "mcp-cli": ("0.1.0", "0.3.0"),
        "mcp-server": ("0.1.0", "0.3.0")
    }
    
    # Check each dependency
    for package, (min_version, max_version) in dependencies.items():
        try:
            compatible, error = check_dependency_version(package, min_version, max_version)
            if not compatible:
                issues.append(error)
        except Exception as e:
            issues.append(f"Error checking {package}: {str(e)}")
    
    # Special check for NumPy to ensure it's not v2.x
    try:
        import numpy
        numpy_version = numpy.__version__
        if numpy_version.startswith("2."):
            issues.append(f"NumPy version {numpy_version} is not supported. Please use NumPy 1.x (e.g., 1.20.0 or higher).")
    except ImportError:
        # Already reported by the dependency check
        pass
    
    return CompatibilityReport(
        compatible=len(issues) == 0,
        issues=issues,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )


def print_compatibility_report(report: CompatibilityReport) -> None:
    """
    Print a compatibility report to the logger.
    
    Args:
        report: The compatibility report to print
    """
    if report.compatible:
        logger.info(f"Environment is compatible (Python {report.python_version})")
    else:
        logger.error(f"Environment has compatibility issues (Python {report.python_version}):")
        for issue in report.issues:
            logger.error(f"  - {issue}")
        
        # Print helpful message
        logger.info("To resolve these issues, you can try:")
        logger.info("  - Use Python 3.8-3.12")
        logger.info("  - Install dependencies with: pip install -r requirements.txt")
        logger.info("  - If using NumPy 2.x, downgrade with: pip install \"numpy>=1.20.0,<2.0.0\"")
