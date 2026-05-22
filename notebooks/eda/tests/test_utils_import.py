"""Test that utils package can be imported safely without ImportError."""

import pytest
import sys
import os

# Add notebooks/eda to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_utils_package_imports_safely():
    """Test that utils package can be imported without ImportError."""
    try:
        import utils
        # Should be able to access version without ImportError
        version = getattr(utils, '__version__', None)
        assert version == "1.0.0"
    except ImportError as e:
        pytest.fail(f"Utils package should import safely but got ImportError: {e}")


def test_utils_package_has_version():
    """Test that utils package has correct version."""
    import utils
    assert hasattr(utils, '__version__')
    assert utils.__version__ == "1.0.0"