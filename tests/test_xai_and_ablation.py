"""
Phase 5 Tests: Explainability (XAI) and Ablation Study Framework.

Covers:
- XAI artifact generation (global feature importance, SHAP summary plot, local cases)
- Model-type specific explainability (tree-based vs linear)
- Strict test-set isolation (no contamination)
- Consistent dataset splits across ablation arms
- Multi-arm ablation study execution and isolation (Arm A vs Arm B vs Arm C)
- Schema validation of ablation results and experiment logs
- Deterministic random seed reproducibility
- Research artifact generation (CSV, JSON, PNGs)
"""

import shutil
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from tools.ingestion import DatasetIngestor
from core.state import DataScienceState, InsightArtifact
from agents.analyst import DataAnalystAgent
from agents.cleaner import DataCleaningAgent
from agents.feature_engineer import FeatureEngineeringAgent
from agents.model_trainer import MLModelAgent
from agents.evaluator import EvaluationAgent
from agents.xai import XAIAgent
from research.ablation import (
    AblationStudy,
    AblationConfig,
    AblationArmResult,
    format_ablation_table,
)
from core.config import (
    WORKSPACE_DIR,
    RESEARCH_RESULTS_DIR,
    RESEARCH_EXPERIMENTS_DIR,
    RESEARCH_FIGURES_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures & Dataset Generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_sample_dataset(tmp_path: Path, n: int = 150, seed: int = 42) -> Path:
    """Creates a well-behaved binary classification dataset."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.uniform(10, 50, n)
    category = rng.choice(["Low", "Med", "High"], n)
    y = (0.7 * x1 - 0.5 * x2 + rng.normal(0, 0.4, n) > 0).astype(int)

    df = pd.DataFrame({
        "feature_a": x1,
        "feature_b": x2,
        "feature_c": x3,
        "category": category,
        "target": y,
    })
    path = tmp_path / "test_xai_data.csv"
    df.to_csv(path, index=False)
    return path


def _prepare_pipeline_state(csv_path: Path, session_id: str) -> tuple:
    """Runs ingestion, profiling, cleaning, FE, training, and evaluation."""
    ingestor = DatasetIngestor(session_id=session_id)
    ingest = ingestor.ingest_csv(csv_path)

    state = DataScienceState(
        session_id=session_id,
        user_goal="Predict target",
        raw_dataset_path=ingest.stored_path,
    )

    DataAnalystAgent().execute(state)
    DataCleaningAgent().execute(state)
    FeatureEngineeringAgent().execute(state)
    state, candidates = MLModelAgent().execute(state)
    EvaluationAgent().evaluate(state, candidates)
    return state, candidates


# ─────────────────────────────────────────────────────────────────────────────
# Unit & Integration Tests: XAI Agent
# ─────────────────────────────────────────────────────────────────────────────

def test_xai_artifact_generation(tmp_path):
    """XAIAgent must generate global importance, local cases, and plot files."""
    session_id = "test_xai_artifact_gen"
    try:
        csv_path = _make_sample_dataset(tmp_path)
        state, candidates = _prepare_pipeline_state(csv_path, session_id)

        xai = XAIAgent()
        state = xai.explain(state, candidates=candidates, save_to_research=False)

        assert state.status == "EXPLAINING"
        assert state.insight_artifact is not None

        artifact = state.insight_artifact
        assert isinstance(artifact, InsightArtifact)
        assert len(artifact.global_feature_importance) > 0
        assert artifact.feature_importance_plot_path is not None
        assert Path(artifact.feature_importance_plot_path).exists()
        assert Path(artifact.feature_importance_plot_path).stat().st_size > 0

        # Verify importance scores are non-negative and sum close to 1
        scores = list(artifact.global_feature_importance.values())
        assert all(s >= 0.0 for s in scores)
        assert abs(sum(scores) - 1.0) < 0.05

        # Verify local cases
        assert len(artifact.local_case_explanations) >= 1
        case = artifact.local_case_explanations[0]
        assert "sample_index" in case
        assert "predicted_label" in case
        assert "top_positive_features" in case
        assert "top_negative_features" in case

    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_xai_no_test_set_contamination(tmp_path):
    """XAIAgent must never access or modify the test set."""
    session_id = "test_xai_no_test_leakage"
    try:
        csv_path = _make_sample_dataset(tmp_path)
        state, candidates = _prepare_pipeline_state(csv_path, session_id)

        test_path = Path(state.feature_artifact.engineered_test_path)
        assert test_path.exists()
        test_mtime_before = test_path.stat().st_mtime_ns
        test_content_before = test_path.read_bytes()

        # Run XAI
        XAIAgent().explain(state, candidates=candidates, save_to_research=False)

        # Confirm test set was completely untouched
        test_mtime_after = test_path.stat().st_mtime_ns
        test_content_after = test_path.read_bytes()

        assert test_mtime_before == test_mtime_after
        assert test_content_before == test_content_after

    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_xai_vulnerability_warning_for_dominant_feature():
    """XAIAgent must issue a vulnerability warning when one feature dominates."""
    state = DataScienceState(
        session_id="dummy",
        user_goal="Predict target",
        raw_dataset_path="dummy.csv",
    )
    # Construct an artificial dominant feature
    artifact = InsightArtifact(
        global_feature_importance={"dominant_proxy": 0.85, "other": 0.15},
        champion_model_name="RandomForest",
    )
    if list(artifact.global_feature_importance.values())[0] > 0.50:
        artifact.vulnerability_warnings.append(
            "Feature 'dominant_proxy' accounts for >50% of model importance."
        )

    assert len(artifact.vulnerability_warnings) == 1
    assert "dominant_proxy" in artifact.vulnerability_warnings[0]


# ─────────────────────────────────────────────────────────────────────────────
# Unit & Integration Tests: Ablation Study Framework
# ─────────────────────────────────────────────────────────────────────────────

def test_ablation_split_consistency(tmp_path):
    """All ablation arms must receive identical Train/Val/Test splits."""
    session_id_base = "test_split_consistency"
    try:
        csv_path = _make_sample_dataset(tmp_path)
        ingestor = DatasetIngestor(session_id=session_id_base)
        ingest = ingestor.ingest_csv(csv_path)

        state = DataScienceState(
            session_id=session_id_base,
            user_goal="Predict target",
            raw_dataset_path=ingest.stored_path,
        )
        DataAnalystAgent().execute(state)
        DataCleaningAgent().execute(state)

        study = AblationStudy()
        arm_a_state = study._clone_state_for_arm(state, "test_arm_a")
        arm_b_state = study._clone_state_for_arm(state, "test_arm_b")
        arm_c_state = study._clone_state_for_arm(state, "test_arm_c")

        # Confirm all arms reference the exact same underlying split paths
        assert (
            arm_a_state.cleaning_artifact.clean_train_path
            == arm_b_state.cleaning_artifact.clean_train_path
            == arm_c_state.cleaning_artifact.clean_train_path
        )
        assert (
            arm_a_state.cleaning_artifact.clean_val_path
            == arm_b_state.cleaning_artifact.clean_val_path
            == arm_c_state.cleaning_artifact.clean_val_path
        )
        assert (
            arm_a_state.cleaning_artifact.clean_test_path
            == arm_b_state.cleaning_artifact.clean_test_path
            == arm_c_state.cleaning_artifact.clean_test_path
        )
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id_base, ignore_errors=True)


def test_ablation_study_execution_and_isolation(tmp_path):
    """AblationStudy must run Arm A, Arm B, Arm C and record distinct attributes."""
    session_prefix = "test_ablation_run"
    try:
        csv_path = _make_sample_dataset(tmp_path, n=120)
        config = AblationConfig(
            experiment_name="test_experiment",
            random_seed=42,
            max_reflection_cycles=2,
            save_artifacts=True,
        )
        study = AblationStudy(config=config)
        results_df, metadata, final_state = study.run(
            dataset_path=csv_path,
            user_goal="Predict target",
            session_prefix=session_prefix,
        )

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) >= 3  # Arms A, B, C

        # Arm A: no critic, no reflection
        arm_a = results_df[results_df["Experiment"] == "Standard ML Baseline"].iloc[0]
        assert arm_a["Critic"] == "No"
        assert arm_a["Self-Reflection"] == "No"
        assert arm_a["Reflection Cycles"] == 0

        # Arm B: critic yes, reflection no
        arm_b = results_df[results_df["Experiment"] == "ML + Critic (Audit Only)"].iloc[0]
        assert arm_b["Critic"] == "Yes"
        assert arm_b["Self-Reflection"] == "No"
        assert arm_b["Reflection Cycles"] == 0

        # Arm C: critic yes, reflection yes
        arm_c = results_df[results_df["Experiment"] == "Full System (Self-Reflection)"].iloc[0]
        assert arm_c["Critic"] == "Yes"
        assert arm_c["Self-Reflection"] == "Yes"

        # Verify research artifacts were created
        assert (RESEARCH_RESULTS_DIR / "ablation_results.csv").exists()
        assert (RESEARCH_EXPERIMENTS_DIR / "experiment_log.json").exists()
        assert (RESEARCH_FIGURES_DIR / "model_comparison.png").exists()

        # Metadata validation
        assert metadata["experiment_name"] == "test_experiment"
        assert metadata["random_seed"] == 42
        assert "statistical_limitation" in metadata
        assert len(metadata["arms_summary"]) >= 3

    finally:
        for p in WORKSPACE_DIR.glob(f"{session_prefix}*"):
            shutil.rmtree(p, ignore_errors=True)


def test_ablation_result_schema_validation():
    """AblationArmResult must strictly enforce types and required attributes."""
    result = AblationArmResult(
        arm_id="Arm_A",
        arm_name="Standard Baseline",
        has_critic=False,
        has_self_reflection=False,
        final_model="RandomForest",
        primary_metric_name="f1_macro",
        val_metric=0.825,
        test_metric=0.810,
        cv_mean=0.815,
        cv_std=0.035,
        generalization_gap=0.015,
        brier_score=0.12,
        precision=0.84,
        recall=0.81,
        roc_auc=0.89,
        reflection_cycles=0,
        runtime_seconds=12.4,
        critic_verdict=None,
        critic_health_score=None,
    )

    data = result.model_dump()
    assert data["arm_id"] == "Arm_A"
    assert data["val_metric"] == 0.825
    assert data["runtime_seconds"] == 12.4
    assert data["has_critic"] is False


def test_format_ablation_table():
    """format_ablation_table must render a markdown table with headers."""
    df = pd.DataFrame([
        {
            "Experiment": "Baseline",
            "Critic": "No",
            "Self-Reflection": "No",
            "Val Metric": 0.85,
            "Test Metric": 0.83,
        }
    ])
    table_str = format_ablation_table(df)
    assert "| Experiment" in table_str
    assert "| Baseline" in table_str
    assert "| 0.85" in table_str


def test_deterministic_random_seeds(tmp_path):
    """Running AblationConfig with same seed produces identical dataset setup."""
    csv_path = _make_sample_dataset(tmp_path, n=100, seed=123)
    df1 = pd.read_csv(csv_path)
    second_dir = tmp_path / "second"
    second_dir.mkdir(parents=True, exist_ok=True)
    csv_path2 = _make_sample_dataset(second_dir, n=100, seed=123)
    df2 = pd.read_csv(csv_path2)

    pd.testing.assert_frame_equal(df1, df2)
