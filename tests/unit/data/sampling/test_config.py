"""
Unit tests for sampling configuration module.

Tests the SamplingConfig dataclass that defines parameters for M5 dataset
behavioral stratification and sampling. This configuration enables creating
statistically rigorous sample datasets while maintaining behavioral diversity
across volume, intermittency, and lifecycle dimensions.
"""

import pytest
from demand_forecast_intelligence.data.sampling.config import SamplingConfig


class TestSamplingConfigDefaults:
    """Test default configuration values are correctly set."""

    def test_creates_config_with_correct_defaults(self):
        """Test that SamplingConfig initializes with expected default values."""
        config = SamplingConfig()

        # Core sampling parameters - designed for 40-50% dataset reduction
        assert config.target_item_count == 1400, "Should target ~46% of 3,049 M5 items"
        assert config.random_seed == 42, "Should use standard seed for reproducibility"
        assert config.training_end_day == "d_1913", "Must prevent future leakage"

        # Behavioral stratification thresholds - statistical quintiles + extremes
        assert config.volume_percentiles == [0, 25, 75, 95, 100], "Should create 4 volume strata"
        assert config.intermittency_thresholds == [0.2, 0.6], "Should create 3 intermittency levels"

        # Lifecycle thresholds - temporal classification
        expected_lifecycle = {
            "early_end": "d_1000",
            "late_start": "d_1000",
            "longrun_min_span": 900,
            "discontinued_cutoff": "d_1700"
        }
        assert config.lifecycle_thresholds == expected_lifecycle, "Should have correct lifecycle defaults"

        # Business constraints - ensure department representation
        assert config.min_per_dept == 30, "Should ensure adequate dept representation"
        assert config.min_per_stratum == 2, "Should prevent empty strata"


class TestSamplingConfigValidation:
    """Test input validation in __post_init__ method."""

    def test_rejects_negative_target_item_count(self):
        """Test that negative target_item_count raises ValueError."""
        with pytest.raises(ValueError, match="target_item_count must be positive"):
            SamplingConfig(target_item_count=-1)

    def test_rejects_zero_target_item_count(self):
        """Test that zero target_item_count raises ValueError."""
        with pytest.raises(ValueError, match="target_item_count must be positive"):
            SamplingConfig(target_item_count=0)

    def test_rejects_invalid_training_end_day_format(self):
        """Test that training_end_day must follow d_XXXX format."""
        with pytest.raises(ValueError, match="training_end_day must be in format 'd_XXXX'"):
            SamplingConfig(training_end_day="day_1913")

    def test_rejects_training_end_day_without_d_prefix(self):
        """Test that training_end_day requires 'd_' prefix."""
        with pytest.raises(ValueError, match="training_end_day must be in format 'd_XXXX'"):
            SamplingConfig(training_end_day="1913")

    def test_rejects_empty_volume_percentiles(self):
        """Test that volume_percentiles cannot be empty."""
        with pytest.raises(ValueError, match="volume_percentiles cannot be empty"):
            SamplingConfig(volume_percentiles=[])

    def test_rejects_unsorted_volume_percentiles(self):
        """Test that volume_percentiles must be in ascending order."""
        with pytest.raises(ValueError, match="volume_percentiles must be sorted in ascending order"):
            SamplingConfig(volume_percentiles=[25, 0, 75])

    def test_rejects_volume_percentiles_outside_range(self):
        """Test that volume_percentiles must be between 0 and 100."""
        with pytest.raises(ValueError, match="volume_percentiles must be between 0 and 100"):
            SamplingConfig(volume_percentiles=[0, 25, 105])

    def test_rejects_negative_volume_percentiles(self):
        """Test that volume_percentiles cannot be negative."""
        with pytest.raises(ValueError, match="volume_percentiles must be between 0 and 100"):
            SamplingConfig(volume_percentiles=[-5, 25, 75])

    def test_rejects_empty_intermittency_thresholds(self):
        """Test that intermittency_thresholds cannot be empty."""
        with pytest.raises(ValueError, match="intermittency_thresholds cannot be empty"):
            SamplingConfig(intermittency_thresholds=[])

    def test_rejects_unsorted_intermittency_thresholds(self):
        """Test that intermittency_thresholds must be in ascending order."""
        with pytest.raises(ValueError, match="intermittency_thresholds must be sorted in ascending order"):
            SamplingConfig(intermittency_thresholds=[0.6, 0.2])

    def test_rejects_intermittency_thresholds_outside_range(self):
        """Test that intermittency_thresholds must be between 0 and 1."""
        with pytest.raises(ValueError, match="intermittency_thresholds must be between 0 and 1"):
            SamplingConfig(intermittency_thresholds=[0.2, 1.5])

    def test_rejects_negative_intermittency_thresholds(self):
        """Test that intermittency_thresholds cannot be negative."""
        with pytest.raises(ValueError, match="intermittency_thresholds must be between 0 and 1"):
            SamplingConfig(intermittency_thresholds=[-0.1, 0.2])

    def test_rejects_negative_min_per_dept(self):
        """Test that min_per_dept must be positive."""
        with pytest.raises(ValueError, match="min_per_dept must be positive"):
            SamplingConfig(min_per_dept=-1)

    def test_rejects_zero_min_per_dept(self):
        """Test that min_per_dept must be positive."""
        with pytest.raises(ValueError, match="min_per_dept must be positive"):
            SamplingConfig(min_per_dept=0)

    def test_rejects_negative_min_per_stratum(self):
        """Test that min_per_stratum must be positive."""
        with pytest.raises(ValueError, match="min_per_stratum must be positive"):
            SamplingConfig(min_per_stratum=-1)

    def test_rejects_zero_min_per_stratum(self):
        """Test that min_per_stratum must be positive."""
        with pytest.raises(ValueError, match="min_per_stratum must be positive"):
            SamplingConfig(min_per_stratum=0)

    def test_rejects_negative_random_seed(self):
        """Test that random_seed must be non-negative."""
        with pytest.raises(ValueError, match="random_seed must be non-negative"):
            SamplingConfig(random_seed=-1)

    def test_rejects_invalid_lifecycle_early_end_format(self):
        """Test that lifecycle_thresholds early_end must follow d_XXXX format."""
        with pytest.raises(ValueError, match="lifecycle_thresholds\['early_end'\] must be in format 'd_XXXX'"):
            SamplingConfig(lifecycle_thresholds={
                "early_end": "day_1000",
                "late_start": "d_1000",
                "longrun_min_span": 900,
                "discontinued_cutoff": "d_1700"
            })

    def test_rejects_invalid_lifecycle_late_start_format(self):
        """Test that lifecycle_thresholds late_start must follow d_XXXX format."""
        with pytest.raises(ValueError, match="lifecycle_thresholds\['late_start'\] must be in format 'd_XXXX'"):
            SamplingConfig(lifecycle_thresholds={
                "early_end": "d_1000",
                "late_start": "1000",
                "longrun_min_span": 900,
                "discontinued_cutoff": "d_1700"
            })

    def test_rejects_invalid_lifecycle_discontinued_cutoff_format(self):
        """Test that lifecycle_thresholds discontinued_cutoff must follow d_XXXX format."""
        with pytest.raises(ValueError, match="lifecycle_thresholds\['discontinued_cutoff'\] must be in format 'd_XXXX'"):
            SamplingConfig(lifecycle_thresholds={
                "early_end": "d_1000",
                "late_start": "d_1000",
                "longrun_min_span": 900,
                "discontinued_cutoff": "discontinued_1700"
            })

    def test_rejects_negative_lifecycle_longrun_min_span(self):
        """Test that lifecycle_thresholds longrun_min_span must be positive."""
        with pytest.raises(ValueError, match="lifecycle_thresholds\['longrun_min_span'\] must be positive"):
            SamplingConfig(lifecycle_thresholds={
                "early_end": "d_1000",
                "late_start": "d_1000",
                "longrun_min_span": -100,
                "discontinued_cutoff": "d_1700"
            })

    def test_rejects_missing_lifecycle_threshold_keys(self):
        """Test that lifecycle_thresholds must contain all required keys."""
        with pytest.raises(ValueError, match="lifecycle_thresholds must contain all required keys"):
            SamplingConfig(lifecycle_thresholds={
                "early_end": "d_1000",
                "late_start": "d_1000"
                # Missing longrun_min_span and discontinued_cutoff
            })


class TestSamplingConfigValidInput:
    """Test that valid inputs are accepted without errors."""

    def test_accepts_valid_custom_configuration(self):
        """Test that valid custom configuration values are accepted."""
        custom_lifecycle = {
            "early_end": "d_800",
            "late_start": "d_900",
            "longrun_min_span": 500,
            "discontinued_cutoff": "d_1500"
        }
        config = SamplingConfig(
            target_item_count=1000,
            random_seed=123,
            training_end_day="d_1800",
            volume_percentiles=[10, 50, 90],
            intermittency_thresholds=[0.3, 0.7],
            lifecycle_thresholds=custom_lifecycle,
            min_per_dept=20,
            min_per_stratum=1
        )

        assert config.target_item_count == 1000
        assert config.random_seed == 123
        assert config.training_end_day == "d_1800"
        assert config.volume_percentiles == [10, 50, 90]
        assert config.intermittency_thresholds == [0.3, 0.7]
        assert config.lifecycle_thresholds == custom_lifecycle
        assert config.min_per_dept == 20
        assert config.min_per_stratum == 1

    def test_accepts_single_volume_percentile(self):
        """Test that single volume percentile is valid."""
        config = SamplingConfig(volume_percentiles=[50])
        assert config.volume_percentiles == [50]

    def test_accepts_single_intermittency_threshold(self):
        """Test that single intermittency threshold is valid."""
        config = SamplingConfig(intermittency_thresholds=[0.5])
        assert config.intermittency_thresholds == [0.5]

    def test_accepts_boundary_values(self):
        """Test that boundary values (0, 1, 100) are accepted."""
        config = SamplingConfig(
            volume_percentiles=[0, 100],
            intermittency_thresholds=[0.0, 1.0]
        )
        assert config.volume_percentiles == [0, 100]
        assert config.intermittency_thresholds == [0.0, 1.0]

    def test_accepts_zero_random_seed(self):
        """Test that zero random_seed is valid."""
        config = SamplingConfig(random_seed=0)
        assert config.random_seed == 0

    def test_accepts_custom_lifecycle_thresholds(self):
        """Test that custom lifecycle threshold values are accepted."""
        custom_lifecycle = {
            "early_end": "d_500",
            "late_start": "d_1200",
            "longrun_min_span": 1000,
            "discontinued_cutoff": "d_1900"
        }
        config = SamplingConfig(lifecycle_thresholds=custom_lifecycle)
        assert config.lifecycle_thresholds == custom_lifecycle


class TestSamplingConfigBusinessLogic:
    """Test business logic and derived properties."""

    def test_volume_strata_count_calculation(self):
        """Test that number of volume strata is calculated correctly."""
        config = SamplingConfig(volume_percentiles=[0, 25, 75, 95, 100])
        # 5 percentiles create 4 strata: [0-25), [25-75), [75-95), [95-100]
        expected_strata = len(config.volume_percentiles) - 1
        assert expected_strata == 4, "Should create 4 volume strata"

    def test_intermittency_levels_count_calculation(self):
        """Test that number of intermittency levels is calculated correctly."""
        config = SamplingConfig(intermittency_thresholds=[0.2, 0.6])
        # 2 thresholds create 3 levels: [0-0.2), [0.2-0.6), [0.6-1.0]
        expected_levels = len(config.intermittency_thresholds) + 1
        assert expected_levels == 3, "Should create 3 intermittency levels"

    def test_total_behavioral_combinations(self):
        """Test calculation of total behavioral stratification combinations."""
        config = SamplingConfig()
        volume_strata = len(config.volume_percentiles) - 1  # 4 strata
        intermittency_levels = len(config.intermittency_thresholds) + 1  # 3 levels
        total_combinations = volume_strata * intermittency_levels
        assert total_combinations == 12, "Should create 12 behavioral combinations"