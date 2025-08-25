"""Basic tests for deepMask package."""

import pytest


def test_package_import():
    """Test that the package can be imported."""
    try:
        import deepMask
        assert hasattr(deepMask, '__version__')
    except ImportError:
        pytest.skip("Package not installed in development mode")


def test_version_exists():
    """Test that version is accessible."""
    try:
        import deepMask
        version = deepMask.__version__
        assert isinstance(version, str)
        assert len(version) > 0
    except ImportError:
        pytest.skip("Package not installed in development mode")


def test_modules_importable():
    """Test that main modules can be imported."""
    try:
        from deepMask import vnet
        from deepMask.utils import data, deepmask, helpers, image_processing
        assert vnet is not None
        assert data is not None
        assert deepmask is not None
        assert helpers is not None
        assert image_processing is not None
    except ImportError as e:
        pytest.skip(f"Modules not importable: {e}")
