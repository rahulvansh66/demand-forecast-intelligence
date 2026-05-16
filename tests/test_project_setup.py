import importlib.util

def test_package_importable():
    """Test that the main package can be imported."""
    spec = importlib.util.spec_from_file_location(
        "demand_forecast_intelligence",
        "src/demand_forecast_intelligence/__init__.py"
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)