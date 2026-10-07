"""
Unit and integration tests for Feature Engineering, ML Model Agent, and Evaluation Engine.
Verifies leakage boundaries, cross-validation, Optuna tuning, and test-set isolation.
"""

import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from core.state import DataScienceState
from agents.analyst import DataAnalystAgent
from agents.cleaner import DataCleaningAgent
from agents.feature_engineer import FeatureEngineeringAgent
from agents.model_trainer import MLModelAgent
from agents.evaluator import EvaluationAgent
from core.config import WORKSPACE_DIR


@pytest.fixture
def classification_dataset(tmp_path):
    """Generates synthetic classification dataset."""
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "age": np.random.normal(35, 10, n),
        "income": np.random.exponential(40000, n),
        "department": np.random.choice(["Sales", "Engineering", "Marketing", "HR"], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    csv_path = tmp_path / "classification_sample.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def regression_dataset(tmp_path):
    """Generates synthetic regression dataset."""
    np.random.seed(42)
    n = 150
    x1 = np.random.normal(10, 3, n)
    x2 = np.random.exponential(5, n)
    cat = np.random.choice(["Low", "Med", "High"], n)
    y = 2.5 * x1 + 1.2 * x2 + np.random.normal(0, 1, n)

    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "category": cat,
        "target": y
    })
    csv_path = tmp_path / "regression_sample.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_feature_engineering_train_only_fit(classification_dataset):
    """Verifies that feature engineering fits strictly on the train split without leakage."""
    session_id = "test_fe_session"
    state = DataScienceState(
        session_id=session_id,
        user_goal="Predict employee churn target",
        raw_dataset_path=classification_dataset
    )

    DataAnalystAgent().execute(state)
    DataCleaningAgent().execute(state)
    fe_agent = FeatureEngineeringAgent()
    updated_state = fe_agent.execute(state)

    artifact = updated_state.feature_artifact
    assert artifact is not None
    assert artifact.total_features_count > 0
    assert Path(artifact.engineered_train_path).exists()
    assert Path(artifact.engineered_val_path).exists()
    assert Path(artifact.engineered_test_path).exists()

    # Verify scaled features mean ~0 on train
    train_df = pd.read_parquet(artifact.engineered_train_path)
    val_df = pd.read_parquet(artifact.engineered_val_path)
    
    # Train scaled columns should be centered around 0
    assert abs(train_df["age_scaled"].mean()) < 1e-2

    # Clean up workspace
    shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_classification_tournament_and_leaderboard(classification_dataset):
    """Verifies end-to-end model training, CV, Optuna tuning, and leaderboard generation."""
    session_id = "test_clf_tournament"
    state = DataScienceState(
        session_id=session_id,
        user_goal="Predict employee target",
        raw_dataset_path=classification_dataset
    )

    DataAnalystAgent().execute(state)
    DataCleaningAgent().execute(state)
    FeatureEngineeringAgent().execute(state)

    # 1. Model Training
    trainer = MLModelAgent()
    state, candidates = trainer.execute(state)

    assert len(candidates) >= 4  # LogisticRegression, RandomForest, HistGBDT, LightGBM, XGBoost, etc.
    for c in candidates:
        assert "f1_macro" in c.metrics
        metric = c.metrics["f1_macro"]
        assert metric.train_score is not None
        assert metric.val_score is not None
        assert metric.cv_mean is not None
        assert metric.cv_std is not None
        # Verify test score is UNTOUCHED during model training!
        assert metric.test_score is None

    # 2. Evaluation Engine
    evaluator = EvaluationAgent()
    state = evaluator.evaluate(state, candidates)

    report = state.evaluation_report
    assert report is not None
    assert len(report.leaderboard) == len(candidates)
    assert state.champion_model_name is not None
    assert report.champion_model_name == state.champion_model_name

    # Check champion model test score was populated ONLY after champion selection
    champ = next(c for c in report.leaderboard if c.is_champion)
    assert champ.metrics["f1_macro"].test_score is not None
    assert champ.metrics["f1_macro"].bootstrap_ci_lower is not None
    assert champ.metrics["f1_macro"].bootstrap_ci_upper is not None
    assert report.confusion_matrix is not None
    assert report.classification_report is not None

    # Clean up workspace
    shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_regression_tournament_and_residuals(regression_dataset):
    """Verifies end-to-end regression training, residual analysis, and R2 scoring."""
    session_id = "test_reg_tournament"
    state = DataScienceState(
        session_id=session_id,
        user_goal="Predict continuous target value",
        raw_dataset_path=regression_dataset
    )

    DataAnalystAgent().execute(state)
    DataCleaningAgent().execute(state)
    FeatureEngineeringAgent().execute(state)

    trainer = MLModelAgent()
    state, candidates = trainer.execute(state)

    evaluator = EvaluationAgent()
    state = evaluator.evaluate(state, candidates)

    report = state.evaluation_report
    assert report is not None
    assert state.profile.problem_type == "regression"
    assert report.residual_diagnostics is not None
    assert "rmse" in report.residual_diagnostics
    assert "mae" in report.residual_diagnostics
    assert "residual_mean" in report.residual_diagnostics

    # Clean up workspace
    shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_test_set_isolation_guarantee(classification_dataset):
    """Ensures test data is completely isolated and never evaluated until champion is selected."""
    session_id = "test_isolation_check"
    state = DataScienceState(
        session_id=session_id,
        user_goal="Predict target",
        raw_dataset_path=classification_dataset
    )

    DataAnalystAgent().execute(state)
    DataCleaningAgent().execute(state)
    FeatureEngineeringAgent().execute(state)

    trainer = MLModelAgent()
    state, candidates = trainer.execute(state)

    # In candidates returned by trainer, NO model should have an evaluated test score
    for cand in candidates:
        for m_name, m_val in cand.metrics.items():
            assert m_val.test_score is None, f"Model {cand.model_name} accessed test set prematurely!"

    shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)
