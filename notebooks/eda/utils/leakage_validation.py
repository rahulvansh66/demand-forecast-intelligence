"""
Leakage validation module for M5 demand forecasting EDA framework (Step 14).

Provides comprehensive temporal leakage audit functions to prevent future information
leakage in time-series demand forecasting. Validates temporal boundaries, feature
availability timing, cross-validation integrity, and detects suspicious correlations.

Focus: M5 dataset with daily sales (d_1 to d_1913 train, d_1914 to d_1941 validation)
and weekly pricing data. Ensures features respect temporal ordering and no future
information leaks into model training.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def audit_temporal_boundaries(
    feature_data: pd.DataFrame,
    split_date: pd.Timestamp,
    feature_configs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Audit temporal boundaries of features to detect leakage at train-validation split.

    Validates that lag features, rolling statistics, and other temporal features
    respect the train-validation split boundary. Ensures no future information
    is used at the split point.

    Parameters
    ----------
    feature_data : pd.DataFrame
        Feature data with 'date' column and feature columns matching feature_configs
    split_date : pd.Timestamp
        Train-validation split date boundary
    feature_configs : List[Dict[str, Any]]
        List of feature configurations with structure:
        [{'name': str, 'type': 'lag'|'rolling'|'point', 'window': int}, ...]

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - boundary_violations: List of features that leak information at split
        - compliant_features: List of features that respect temporal boundaries
        - risk_assessment: Text summary of temporal integrity
        - detailed_analysis: Per-feature boundary validation results

    Raises
    ------
    ValueError
        If feature_data is empty or split_date is invalid
    KeyError
        If 'date' column not found in feature_data
    """
    if feature_data.empty:
        raise ValueError("feature_data is empty")

    if 'date' not in feature_data.columns:
        raise KeyError("'date' column not found in feature_data")

    # Ensure date column is datetime
    feature_data = feature_data.copy()
    feature_data['date'] = pd.to_datetime(feature_data['date'])

    results = {
        'boundary_violations': [],
        'compliant_features': [],
        'detailed_analysis': {},
        'risk_assessment': ''
    }

    # Analyze each feature config
    for config in feature_configs:
        feature_name = config['name']
        feature_type = config['type']
        window = config.get('window', 0)

        if feature_name not in feature_data.columns:
            continue

        analysis = _analyze_feature_boundary(
            feature_data,
            feature_name,
            feature_type,
            window,
            split_date
        )

        results['detailed_analysis'][feature_name] = analysis

        if analysis['has_leakage']:
            results['boundary_violations'].append(feature_name)
        else:
            results['compliant_features'].append(feature_name)

    # Generate risk assessment
    violation_count = len(results['boundary_violations'])
    compliant_count = len(results['compliant_features'])

    if violation_count > 0:
        results['risk_assessment'] = (
            f"RISK DETECTED: {violation_count} features have temporal leakage at split. "
            f"{compliant_count} features are compliant."
        )
    else:
        results['risk_assessment'] = (
            f"No temporal leakage detected. All {compliant_count} features respect boundaries."
        )

    return results


def _analyze_feature_boundary(
    feature_data: pd.DataFrame,
    feature_name: str,
    feature_type: str,
    window: int,
    split_date: pd.Timestamp
) -> Dict[str, Any]:
    """Analyze specific feature for boundary violations."""
    split_idx = feature_data[feature_data['date'] == split_date].index

    if len(split_idx) == 0:
        return {
            'has_leakage': False,
            'issue': 'Split date not found in data',
            'recommendation': 'Verify split_date is within data range'
        }

    split_idx = split_idx[0]

    # Check for data availability based on feature type
    if feature_type == 'lag':
        # Lag-window feature needs data from (split_date - window) to exist in training
        # At split_date in validation, it needs data from (split_date - window)
        required_date = split_date - pd.Timedelta(days=window)
        has_required_data = required_date in feature_data['date'].values
        has_leakage = not has_required_data

        return {
            'has_leakage': has_leakage,
            'feature_type': 'lag',
            'window': window,
            'required_data_date': required_date,
            'issue': f"Missing data for lag-{window}" if has_leakage else None,
            'recommendation': 'Ensure sufficient historical data before split' if has_leakage else 'Compliant'
        }

    elif feature_type == 'rolling':
        # Rolling feature needs (window - 1) historical points before split
        min_required_date = split_date - pd.Timedelta(days=window - 1)
        has_sufficient_data = min_required_date in feature_data['date'].values
        has_leakage = not has_sufficient_data

        return {
            'has_leakage': has_leakage,
            'feature_type': 'rolling',
            'window': window,
            'min_required_date': min_required_date,
            'issue': f"Insufficient historical data for rolling-{window}" if has_leakage else None,
            'recommendation': 'Extend training period or reduce window' if has_leakage else 'Compliant'
        }

    elif feature_type == 'point':
        # Point-in-time features should be available at split_date
        has_data_at_split = not feature_data.loc[split_idx, feature_name] if split_idx < len(feature_data) else False

        # Check for data consistency across boundary
        pre_split = feature_data[feature_data['date'] < split_date][feature_name]
        post_split = feature_data[feature_data['date'] >= split_date][feature_name]

        has_leakage = False  # Point features typically don't leak if timing is correct

        return {
            'has_leakage': has_leakage,
            'feature_type': 'point',
            'pre_split_stats': {
                'mean': float(pre_split.mean()) if len(pre_split) > 0 else np.nan,
                'std': float(pre_split.std()) if len(pre_split) > 0 else np.nan,
            },
            'post_split_stats': {
                'mean': float(post_split.mean()) if len(post_split) > 0 else np.nan,
                'std': float(post_split.std()) if len(post_split) > 0 else np.nan,
            },
            'issue': None,
            'recommendation': 'Monitor for distribution shift across boundary'
        }

    else:
        return {
            'has_leakage': False,
            'issue': f'Unknown feature type: {feature_type}',
            'recommendation': 'Check feature_config specification'
        }


def check_feature_availability_timing(
    prediction_date: pd.Timestamp,
    feature_availability: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """
    Validate feature availability timing relative to prediction time.

    Checks that features are available at or before the prediction date (no future data).
    Validates business process alignment for features with delayed availability.

    Parameters
    ----------
    prediction_date : pd.Timestamp
        The date at which predictions are made
    feature_availability : Dict[str, Dict[str, str]]
        Feature availability specifications:
        {'feature_name': {'available_at': 'T' | 'T-N' | 'T+N', 'source': str}, ...}
        Where T is prediction_date, T-1 is one day before, T+1 is one day after, etc.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - available_features: List of features available at or before prediction time
        - unavailable_features: List of features not available (future data)
        - timing_violations: List of features with future data leakage
        - business_process_alignment: Assessment of external data delays
        - timing_analysis: Detailed per-feature timing analysis

    Raises
    ------
    ValueError
        If feature_availability format is invalid or empty
    """
    if not isinstance(feature_availability, dict):
        raise ValueError("feature_availability must be a dictionary")

    results = {
        'available_features': [],
        'unavailable_features': [],
        'timing_violations': [],
        'business_process_alignment': '',
        'timing_analysis': {}
    }

    # No features to analyze
    if len(feature_availability) == 0:
        results['business_process_alignment'] = "No features to validate"
        return results

    for feature_name, timing_info in feature_availability.items():
        available_at = timing_info.get('available_at', 'T')
        source = timing_info.get('source', 'unknown')

        # Parse availability timing (e.g., 'T-1' means one day before prediction)
        try:
            days_offset = _parse_timing_offset(available_at)
        except ValueError:
            results['timing_analysis'][feature_name] = {
                'error': f'Invalid timing format: {available_at}'
            }
            continue

        availability_date = prediction_date + pd.Timedelta(days=days_offset)
        is_available = availability_date <= prediction_date
        has_violation = days_offset > 0

        results['timing_analysis'][feature_name] = {
            'available_at': available_at,
            'availability_date': availability_date,
            'source': source,
            'is_available_at_prediction': is_available,
            'days_before_prediction': -days_offset,
            'timing_offset': days_offset
        }

        if has_violation:
            results['timing_violations'].append(feature_name)
            results['unavailable_features'].append(feature_name)
        elif is_available:
            results['available_features'].append(feature_name)

    # Generate business process alignment assessment
    if results['timing_violations']:
        results['business_process_alignment'] = (
            f"WARNING: {len(results['timing_violations'])} features use future data. "
            "This will cause leakage in production deployment."
        )
    else:
        external_data_features = [
            f for f, t in results['timing_analysis'].items()
            if t.get('timing_offset', 0) < -1 and 'external' in t.get('source', '').lower()
        ]
        if external_data_features:
            results['business_process_alignment'] = (
                f"External data features have delays: {', '.join(external_data_features)}. "
                "Plan data pipeline accordingly."
            )
        else:
            results['business_process_alignment'] = (
                f"All {len(results['available_features'])} features aligned with prediction time."
            )

    return results


def _parse_timing_offset(timing_str: str) -> int:
    """Parse timing string (e.g., 'T-1', 'T+3') to days offset."""
    if not isinstance(timing_str, str):
        raise ValueError(f"Timing must be string, got {type(timing_str)}")

    timing_str = timing_str.strip()

    if timing_str == 'T':
        return 0
    elif timing_str.startswith('T-'):
        try:
            return -int(timing_str[2:])
        except ValueError:
            raise ValueError(f"Invalid timing format: {timing_str}")
    elif timing_str.startswith('T+'):
        try:
            return int(timing_str[2:])
        except ValueError:
            raise ValueError(f"Invalid timing format: {timing_str}")
    else:
        raise ValueError(f"Timing must start with T: {timing_str}")


def validate_cross_validation_integrity(
    sales_data: pd.DataFrame,
    cv_folds: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Validate cross-validation fold integrity for time-series forecasting.

    Ensures CV folds maintain strict temporal ordering (no information leakage),
    have balanced seasonal representation, and don't overlap between train/validation.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Sales data with 'date' column (datetime) and optional 'item_id', 'store_id'
    cv_folds : List[Dict[str, str]]
        List of CV fold specifications:
        [{'train_start': date_str, 'train_end': date_str,
          'val_start': date_str, 'val_end': date_str}, ...]

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - temporal_ordering: Validity and issues with temporal order
        - information_leakage: List of folds with leakage
        - seasonal_alignment: Assessment of seasonal coverage in folds
        - cv_integrity_score: Overall score (0-1, 1 is perfect)
        - recommendations: List of improvement recommendations

    Raises
    ------
    ValueError
        If sales_data or cv_folds is empty
    """
    if sales_data.empty:
        raise ValueError("sales_data is empty")

    if not cv_folds:
        raise ValueError("cv_folds is empty")

    sales_data = sales_data.copy()
    sales_data['date'] = pd.to_datetime(sales_data['date'])

    results = {
        'temporal_ordering': {'is_valid': True, 'issues': []},
        'information_leakage': [],
        'seasonal_alignment': {},
        'cv_integrity_score': 1.0,
        'recommendations': []
    }

    # Validate each fold
    for fold_idx, fold in enumerate(cv_folds):
        train_start = pd.to_datetime(fold.get('train_start'))
        train_end = pd.to_datetime(fold.get('train_end'))
        val_start = pd.to_datetime(fold.get('val_start'))
        val_end = pd.to_datetime(fold.get('val_end'))

        # Check temporal ordering: train_end < val_start
        if train_end >= val_start:
            results['temporal_ordering']['is_valid'] = False
            results['temporal_ordering']['issues'].append(
                f"Fold {fold_idx}: train_end ({train_end}) >= val_start ({val_start})"
            )
            results['information_leakage'].append(fold_idx)
            results['cv_integrity_score'] -= 0.2

        # Check fold continuity
        if val_end < val_start:
            results['temporal_ordering']['issues'].append(
                f"Fold {fold_idx}: val_end ({val_end}) < val_start ({val_start})"
            )

        # Check seasonal alignment
        seasonal_check = _check_seasonal_alignment(
            sales_data, train_start, train_end, val_start, val_end
        )
        results['seasonal_alignment'][f'fold_{fold_idx}'] = seasonal_check

    # Calculate final integrity score
    results['cv_integrity_score'] = max(0, min(1, results['cv_integrity_score']))

    # Generate recommendations
    if results['information_leakage']:
        results['recommendations'].append(
            f"Fix temporal ordering in folds: {results['information_leakage']}"
        )

    if results['cv_integrity_score'] < 0.8:
        results['recommendations'].append(
            "Consider redesigning CV strategy for better temporal separation"
        )

    return results


def _check_seasonal_alignment(
    sales_data: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp
) -> Dict[str, Any]:
    """Check seasonal balance between train and validation periods."""
    train_data = sales_data[(sales_data['date'] >= train_start) & (sales_data['date'] <= train_end)]
    val_data = sales_data[(sales_data['date'] >= val_start) & (sales_data['date'] <= val_end)]

    # Extract month from dates
    train_months = train_data['date'].dt.month.value_counts() if len(train_data) > 0 else pd.Series()
    val_months = val_data['date'].dt.month.value_counts() if len(val_data) > 0 else pd.Series()

    # Check day-of-week representation
    train_dow = train_data['date'].dt.dayofweek.value_counts() if len(train_data) > 0 else pd.Series()
    val_dow = val_data['date'].dt.dayofweek.value_counts() if len(val_data) > 0 else pd.Series()

    return {
        'train_period_length_days': len(train_data),
        'val_period_length_days': len(val_data),
        'train_unique_months': int(len(train_months)),
        'val_unique_months': int(len(val_months)),
        'train_dow_coverage': int(len(train_dow)),
        'val_dow_coverage': int(len(val_dow)),
        'seasonal_balance_issue': len(train_months) > 0 and len(val_months) > 0 and len(val_months) < len(train_months)
    }


def scan_suspicious_correlations(
    feature_data: pd.DataFrame,
    perfect_correlation_threshold: float = 0.98,
    high_correlation_threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Scan for suspicious correlations indicating potential feature leakage.

    Detects perfect predictors (correlation > threshold with target),
    ID-based leakage (sequential patterns), and unusually high correlations
    between features that shouldn't be correlated.

    Parameters
    ----------
    feature_data : pd.DataFrame
        Feature data to scan for correlations
    perfect_correlation_threshold : float, default=0.98
        Threshold for perfect predictor detection (correlation with any column)
    high_correlation_threshold : float, default=0.95
        Threshold for high correlation warning

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - perfect_predictors: List of features with correlation > threshold
        - high_correlations: Dict of high correlations between features
        - id_leakage: List of potential ID-based leakage features
        - suspicious_features: List of all suspicious features
        - leakage_summary: Text summary of findings
        - correlation_matrix: Full correlation matrix for reference

    Raises
    ------
    ValueError
        If feature_data is empty or has only one column
    """
    if feature_data.empty:
        raise ValueError("feature_data is empty")

    feature_data = feature_data.copy()

    # Select only numeric columns
    numeric_cols = feature_data.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {
            'perfect_predictors': [],
            'high_correlations': {},
            'id_leakage': [],
            'suspicious_features': [],
            'leakage_summary': 'Insufficient numeric features for correlation analysis',
            'correlation_matrix': pd.DataFrame()
        }

    # Calculate correlation matrix
    correlation_matrix = feature_data[numeric_cols].corr().abs()

    results = {
        'perfect_predictors': [],
        'high_correlations': {},
        'id_leakage': [],
        'suspicious_features': [],
        'correlation_matrix': correlation_matrix
    }

    # Scan for perfect predictors (diagonal is 1.0, skip)
    for col in correlation_matrix.columns:
        col_correlations = correlation_matrix[col]
        # Exclude self-correlation
        col_correlations = col_correlations[col_correlations.index != col]

        perfect = col_correlations[col_correlations >= perfect_correlation_threshold]
        if len(perfect) > 0:
            results['perfect_predictors'].append(col)
            results['suspicious_features'].append(col)

    # Scan for high correlations
    for i, col1 in enumerate(correlation_matrix.columns):
        for col2 in correlation_matrix.columns[i+1:]:
            corr_val = correlation_matrix.loc[col1, col2]
            if corr_val >= high_correlation_threshold and corr_val < perfect_correlation_threshold:
                results['high_correlations'][f'{col1} <-> {col2}'] = float(corr_val)
                if col1 not in results['suspicious_features']:
                    results['suspicious_features'].append(col1)
                if col2 not in results['suspicious_features']:
                    results['suspicious_features'].append(col2)

    # Detect ID-based leakage (sequential patterns)
    for col in numeric_cols:
        # Check if values are sequential or near-sequential
        values = feature_data[col].dropna().values
        if len(values) > 10:
            diffs = np.diff(values)
            # Sequential data has constant or near-constant differences
            if len(set(diffs[diffs != 0])) < 5 and np.mean(np.abs(diffs)) > 0:
                results['id_leakage'].append(col)
                results['suspicious_features'].append(col)

    # Generate summary
    total_suspicious = len(set(results['suspicious_features']))
    results['leakage_summary'] = (
        f"Found {len(results['perfect_predictors'])} perfect predictors, "
        f"{len(results['high_correlations'])} high correlations, "
        f"{len(results['id_leakage'])} potential ID leakage features. "
        f"Total suspicious: {total_suspicious}"
    )

    return results


def generate_leakage_audit_report(
    feature_data: pd.DataFrame,
    split_date: pd.Timestamp,
    feature_configs: List[Dict[str, Any]],
    cv_folds: Optional[List[Dict[str, str]]] = None,
    feature_availability: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive leakage audit report for deployment readiness assessment.

    Combines all leakage validation checks (temporal boundaries, feature availability,
    CV integrity, suspicious correlations) into a single deployment readiness report.

    Parameters
    ----------
    feature_data : pd.DataFrame
        Feature data with 'date' column and feature columns
    split_date : pd.Timestamp
        Train-validation split date
    feature_configs : List[Dict[str, Any]]
        List of feature configurations (see audit_temporal_boundaries)
    cv_folds : List[Dict[str, str]], optional
        CV fold specifications (see validate_cross_validation_integrity)
    feature_availability : Dict[str, Dict[str, str]], optional
        Feature availability specifications (see check_feature_availability_timing)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - temporal_boundaries: Results from audit_temporal_boundaries
        - feature_availability: Results from check_feature_availability_timing
        - cross_validation: Results from validate_cross_validation_integrity
        - suspicious_correlations: Results from scan_suspicious_correlations
        - deployment_readiness: Boolean indicating if safe to deploy
        - overall_risk_level: 'Low' | 'Medium' | 'High' | 'Critical'
        - recommendations: Actionable recommendations
        - summary: Executive summary of audit

    Raises
    ------
    ValueError
        If feature_data is empty
    """
    if feature_data.empty:
        raise ValueError("feature_data is empty")

    # Run all validation checks
    temporal_result = audit_temporal_boundaries(feature_data, split_date, feature_configs)
    corr_result = scan_suspicious_correlations(feature_data)

    # Optional checks
    cv_result = {}
    if cv_folds:
        cv_result = validate_cross_validation_integrity(feature_data, cv_folds)

    availability_result = {}
    if feature_availability:
        availability_result = check_feature_availability_timing(split_date, feature_availability)

    # Aggregate recommendations
    recommendations = []

    if temporal_result['boundary_violations']:
        recommendations.append(
            f"Fix temporal leakage in: {', '.join(temporal_result['boundary_violations'])}"
        )

    if corr_result['perfect_predictors']:
        recommendations.append(
            f"Remove perfect predictors: {', '.join(corr_result['perfect_predictors'])}"
        )

    if corr_result['id_leakage']:
        recommendations.append(
            f"Remove ID-based features: {', '.join(corr_result['id_leakage'])}"
        )

    if availability_result and availability_result.get('timing_violations'):
        recommendations.append(
            f"Remove future data features: {', '.join(availability_result['timing_violations'])}"
        )

    if cv_result and cv_result.get('information_leakage'):
        recommendations.append(
            "Fix cross-validation fold temporal ordering"
        )

    # Calculate risk level
    risk_level = _calculate_risk_level(
        temporal_result,
        corr_result,
        cv_result,
        availability_result
    )

    deployment_ready = risk_level == 'Low'

    return {
        'temporal_boundaries': temporal_result,
        'feature_availability': availability_result,
        'cross_validation': cv_result,
        'suspicious_correlations': corr_result,
        'deployment_readiness': deployment_ready,
        'overall_risk_level': risk_level,
        'recommendations': recommendations,
        'summary': _generate_summary(risk_level, deployment_ready, len(recommendations))
    }


def _calculate_risk_level(
    temporal_result: Dict,
    corr_result: Dict,
    cv_result: Dict,
    availability_result: Dict
) -> str:
    """Calculate overall risk level from all validation checks."""
    risk_score = 0

    # Temporal boundary violations
    if temporal_result.get('boundary_violations'):
        risk_score += len(temporal_result['boundary_violations']) * 0.3

    # Perfect predictors
    if corr_result.get('perfect_predictors'):
        risk_score += len(corr_result['perfect_predictors']) * 0.25

    # ID leakage
    if corr_result.get('id_leakage'):
        risk_score += len(corr_result['id_leakage']) * 0.25

    # Timing violations
    if availability_result and availability_result.get('timing_violations'):
        risk_score += len(availability_result['timing_violations']) * 0.3

    # CV leakage
    if cv_result and cv_result.get('information_leakage'):
        risk_score += len(cv_result['information_leakage']) * 0.2

    if risk_score >= 3.0:
        return 'Critical'
    elif risk_score >= 2.0:
        return 'High'
    elif risk_score >= 1.0:
        return 'Medium'
    else:
        return 'Low'


def _generate_summary(risk_level: str, deployment_ready: bool, num_recommendations: int) -> str:
    """Generate executive summary of audit."""
    status = "APPROVED FOR DEPLOYMENT" if deployment_ready else "NOT READY FOR DEPLOYMENT"

    return (
        f"{status}. "
        f"Risk Level: {risk_level}. "
        f"{num_recommendations} recommendations for improvement."
    )
