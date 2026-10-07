"""
Unit and integration tests for Data Profiler, Analyst Agent, Cleaning Agent, and EDA Agent.
Tests anomaly detection, leakage screening, reproducible splitting, and chart generation.
"""

import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from core.state import DataScienceState
from tools.profiler import DataProfiler
from agents.analyst import DataAnalystAgent
from agents.cleaner import DataCleaningAgent
from agents.eda import EDAAgent
from core.config import WORKSPACE_DIR


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Generates a synthetic dataset with known anomalies, missing values, and leakage."""
    np.random.seed(42)
    n_rows = 200

    # 1. Normal numeric and categorical
    age = np.random.normal(40, 12, n_rows)
    income = np.random.exponential(50000, n_rows)
    education = np.random.choice(["High School", "Bachelor", "Master", "PhD"], n_rows)

    # 2. Target (Imbalanced: ~80% 0, ~20% 1)
    target = np.random.choice([0, 1], n_rows, p=[0.8, 0.2])

    # 3. Injected anomalies
    id_col = [f"USR_{i:04d}" for i in range(n_rows)]  # Pure identifier
    constant_col = [999] * n_rows                     # Constant column
    leaky_col = target * 1.0 + np.random.normal(0, 0.01, n_rows) # Correlated ~0.99
    
    # 4. Missing values
    income[np.random.choice(n_rows, 30, replace=False)] = np.nan

    df = pd.DataFrame({
        "user_id": id_col,
        "age": age,
        "income": income,
        "education": education,
        "constant_feature": constant_col,
        "leaky_feature": leaky_col,
        "target": target
    })

    # Add duplicate rows
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)

    csv_path = tmp_path / "synthetic_audit_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_profiler_detection_capabilities(synthetic_dataset):
    """Verifies that the profiler detects types, missingness, duplicates, and anomalies."""
    df = pd.read_csv(synthetic_dataset)
    profile = DataProfiler.profile_dataframe(df, target_column="target", user_goal="Predict target")

    # Column type detection
    assert profile.columns["age"].inferred_type == "numeric_continuous"
    assert profile.columns["education"].inferred_type == "categorical_nominal"
    assert profile.columns["user_id"].inferred_type == "identifier"

    # Missing values
    assert profile.columns["income"].missing_count >= 30
    assert profile.columns["income"].missing_percentage > 10.0

    # Duplicate rows
    assert profile.duplicate_rows_count == 10

    # Constant column
    assert "constant_feature" in profile.constant_columns

    # Class imbalance
    assert profile.problem_type == "binary_classification"
    assert profile.class_imbalance_ratio is not None
    assert profile.class_imbalance_ratio >= 3.0

    # Leakage detection
    assert "leaky_feature" in profile.potential_leakage_columns
    assert "user_id" in profile.potential_leakage_columns


def test_analyst_agent_state_update(synthetic_dataset):
    """Verifies that DataAnalystAgent properly enriches DataScienceState."""
    session_id = "test_analyst_session"
    state = DataScienceState(
        session_id=session_id,
        user_goal="Predict customer target",
        raw_dataset_path=synthetic_dataset
    )

    agent = DataAnalystAgent()
    updated_state = agent.execute(state)

    assert updated_state.profile is not None
    assert updated_state.status == "PROFILING"
    assert updated_state.profile.target_column == "target"
    assert len(updated_state.execution_logs) > 0


def test_cleaning_agent_train_fit_isolation(synthetic_dataset):
    """Verifies that DataCleaningAgent sanitizes data and strictly isolates splits."""
    session_id = "test_cleaner_session"
    state = DataScienceState(
        session_id=session_id,
        user_goal="Predict customer target",
        raw_dataset_path=synthetic_dataset
    )
    # Profile first
    DataAnalystAgent().execute(state)

    # Clean
    cleaner = DataCleaningAgent()
    updated_state = cleaner.execute(state)

    artifact = updated_state.cleaning_artifact
    assert artifact is not None
    assert artifact.leakage_guard_passed is True
    assert "constant_feature" in artifact.dropped_columns
    assert "user_id" in artifact.dropped_columns

    # Verify split parquet files exist
    assert Path(artifact.clean_train_path).exists()
    assert Path(artifact.clean_val_path).exists()
    assert Path(artifact.clean_test_path).exists()

    # Verify no null values remain in processed training split
    train_df = pd.read_parquet(artifact.clean_train_path)
    assert train_df.isnull().sum().sum() == 0

    # Cleanup session directory
    shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_eda_agent_execution_and_charts(synthetic_dataset):
    """Verifies that EDAAgent computes metrics and saves visual charts."""
    session_id = "test_eda_session"
    state = DataScienceState(
        session_id=session_id,
        user_goal="Predict customer target",
        raw_dataset_path=synthetic_dataset
    )

    # Run pipeline up to EDA
    DataAnalystAgent().execute(state)
    DataCleaningAgent().execute(state)
    eda_agent = EDAAgent()
    updated_state = eda_agent.execute(state)

    eda = updated_state.eda_results
    assert eda is not None
    assert len(eda.top_correlations) > 0
    assert len(eda.mutual_info_scores) > 0
    assert len(eda.generated_chart_paths) > 0

    # Verify chart image files were created on disk
    for chart_path in eda.generated_chart_paths:
        assert Path(chart_path).exists()
        assert Path(chart_path).stat().st_size > 0

    # Cleanup session directory
    shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)
