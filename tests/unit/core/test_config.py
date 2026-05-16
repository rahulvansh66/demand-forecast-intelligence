import pytest
from pathlib import Path
import tempfile
import yaml

from demand_forecast_intelligence.core.config.settings import Settings, load_config, get_default_config

def test_load_config_from_yaml():
    """Test loading configuration from YAML file."""
    test_config = {
        'dataset': {
            'name': 'test_dataset',
            'description': 'Test dataset description',
            'paths': {
                'raw': 'test/raw',
                'interim': 'test/interim',
                'processed': 'test/processed'
            },
            'files': {
                'test_file': 'test.csv'
            },
            'validation': {
                'max_missing_percentage': 0.1,
                'min_rows': 100,
                'required_columns': {
                    'test_file': ['id', 'name']
                }
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_path = f.name

    try:
        settings = load_config(temp_path)
        assert settings.dataset.name == 'test_dataset'
        assert settings.dataset.paths.raw == 'test/raw'
    finally:
        Path(temp_path).unlink()

def test_settings_validation():
    """Test that Settings validates required fields."""
    with pytest.raises(ValueError):
        Settings(dataset=None)


def test_load_default_config():
    """Test loading the default project configuration."""
    settings = get_default_config()

    # Verify dataset configuration
    assert settings.dataset.name == "walmart_m5"
    assert "sales_train_validation.csv" in settings.dataset.files.values()
    assert settings.dataset.paths.raw == "data/raw"

    # Verify validation rules
    assert settings.dataset.validation.max_missing_percentage == 0.05
    assert "id" in settings.dataset.validation.required_columns["sales_train_validation"]