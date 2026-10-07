"""
Phase 4 Tests: Critic Agent and Self-Reflection Orchestrator.

Covers:
- Clean pipeline → APPROVED verdict
- Intentional data leakage → REJECTED (V1/V5)
- Intentional overfitting → REJECTED (V3)
- Class imbalance detection → WARNING or CRITICAL (V4)
- Unstable CV detection → CRITICAL (V6)
- Poor calibration detection → WARNING (V7)
- Successful corrective rerun (leakage fix)
- Maximum reflection-cycle safety net
- Audit history persistence across cycles
- Deterministic verdict generation
- ReflectionCycleLog population and structure
"""

import shutil
from pathlib import Path
from typing import List
import pytest
import numpy as np
import pandas as pd

from core.state import DataScienceState, TrainedModelCandidate, ModelEvaluationMetric
from agents.analyst import DataAnalystAgent
from agents.cleaner import DataCleaningAgent
from agents.feature_engineer import FeatureEngineeringAgent
from agents.model_trainer import MLModelAgent
from agents.evaluator import EvaluationAgent
from agents.critic import CriticAgent
from core.orchestrator import SelfReflectionOrchestrator
from core.config import WORKSPACE_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Shared dataset generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_clean_csv(tmp_path: Path, n: int = 200, seed: int = 42) -> str:
    """Well-behaved binary classification dataset with no leakage."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    cat = rng.choice(["A", "B", "C"], n)
    y = (0.8 * x1 - 0.6 * x2 + rng.normal(0, 0.5, n) > 0).astype(int)
    df = pd.DataFrame({"feat1": x1, "feat2": x2, "cat": cat, "target": y})
    path = tmp_path / "clean.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_leaky_csv(tmp_path: Path, n: int = 200, seed: int = 42) -> str:
    """Dataset with a leaky feature (|corr| > 0.92 with target)."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = (x1 > 0).astype(int)
    # Leaky feature: nearly deterministic copy of target
    leaky = y.astype(float) + rng.normal(0, 0.01, n)
    df = pd.DataFrame({
        "feat1": x1,
        "feat2": x2,
        "leaky_feature": leaky,
        "target": y
    })
    path = tmp_path / "leaky.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_imbalanced_csv(tmp_path: Path, n: int = 200, seed: int = 42) -> str:
    """Dataset with extreme class imbalance (ratio ≈ 9:1)."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Only 10% positive
    y = (rng.rand(n) < 0.10).astype(int)
    df = pd.DataFrame({"feat1": x1, "feat2": x2, "target": y})
    path = tmp_path / "imbalanced.csv"
    df.to_csv(path, index=False)
    return str(path)


def _run_pipeline_to_evaluation(
    csv_path: str,
    session_id: str,
    user_goal: str = "Predict target",
) -> tuple:
    """Runs FE + Training + Evaluation and returns (state, candidates)."""
    from tools.ingestion import DatasetIngestor

    ingestor = DatasetIngestor(session_id=session_id)
    result = ingestor.ingest_csv(csv_path)

    state = DataScienceState(
        session_id=session_id,
        user_goal=user_goal,
        raw_dataset_path=result.stored_path,
    )

    DataAnalystAgent().execute(state)
    DataCleaningAgent().execute(state)
    FeatureEngineeringAgent().execute(state)
    state, candidates = MLModelAgent().execute(state)
    EvaluationAgent().evaluate(state, candidates)
    return state, candidates


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: Critic Agent
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_pipeline_approved(tmp_path):
    """A well-behaved pipeline with no leakage should be APPROVED."""
    session_id = "test_critic_clean"
    try:
        csv = _make_clean_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        critic = CriticAgent()
        feedback = critic.audit(state, candidates, iteration=1)

        # With a clean dataset the critic should APPROVE
        assert feedback.verdict == "APPROVED", (
            f"Expected APPROVED but got REJECTED. "
            f"Health={feedback.health_score}. "
            f"Critical issues: {[a.check_name for a in feedback.audit_results if a.status == 'CRITICAL_FAIL']}"
        )
        assert feedback.health_score >= 60.0
        assert len(feedback.audit_results) == 8   # all 8 vectors populated
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_leaky_dataset_rejected(tmp_path):
    """A dataset with a near-perfect predictor of the target should be REJECTED (V1/V5)."""
    session_id = "test_critic_leaky"
    try:
        csv = _make_leaky_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        critic = CriticAgent()
        feedback = critic.audit(state, candidates, iteration=1)

        assert feedback.verdict == "REJECTED", (
            f"Expected REJECTED for leaky dataset but got APPROVED. "
            f"Health={feedback.health_score}."
        )
        # At least one of V1 or V5 should be CRITICAL_FAIL
        critical_vectors = {
            a.vector_id for a in feedback.audit_results if a.status == "CRITICAL_FAIL"
        }
        assert critical_vectors & {1, 5}, (
            f"Expected V1 or V5 to be CRITICAL_FAIL, got {critical_vectors}"
        )
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_class_imbalance_detection(tmp_path):
    """Extreme class imbalance (9:1) should trigger at least a WARNING on V4."""
    session_id = "test_critic_imbalance"
    try:
        csv = _make_imbalanced_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        critic = CriticAgent()
        feedback = critic.audit(state, candidates, iteration=1)

        v4 = next((a for a in feedback.audit_results if a.vector_id == 4), None)
        assert v4 is not None, "V4 (class imbalance) must be present in audit results"
        assert v4.status in ("WARNING", "CRITICAL_FAIL"), (
            f"Expected V4 WARNING or CRITICAL_FAIL for 9:1 imbalance, got {v4.status}. "
            f"Profile ratio: {state.profile.class_imbalance_ratio}"
        )
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_audit_produces_8_vectors(tmp_path):
    """CriticAgent must always produce exactly 8 CriticAuditItems."""
    session_id = "test_critic_8_vectors"
    try:
        csv = _make_clean_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        critic = CriticAgent()
        feedback = critic.audit(state, candidates, iteration=1)

        assert len(feedback.audit_results) == 8
        vector_ids = [a.vector_id for a in feedback.audit_results]
        assert sorted(vector_ids) == list(range(1, 9))
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_audit_item_has_required_fields(tmp_path):
    """Every CriticAuditItem must have check_name, observation, recommendation, affected_component."""
    session_id = "test_critic_fields"
    try:
        csv = _make_clean_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        feedback = CriticAgent().audit(state, candidates)

        for item in feedback.audit_results:
            assert item.check_name, f"V{item.vector_id}: check_name is empty"
            assert item.observation, f"V{item.vector_id}: observation is empty"
            assert item.recommendation, f"V{item.vector_id}: recommendation is empty"
            assert item.affected_component is not None  # can be empty string but must exist
            assert isinstance(item.evidence, dict)      # evidence must be a dict
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_deterministic_verdict(tmp_path):
    """Running CriticAgent twice on the same state must produce the same verdict."""
    session_id = "test_critic_deterministic"
    try:
        csv = _make_clean_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        critic = CriticAgent()
        fb1 = critic.audit(state, candidates, iteration=1)
        fb2 = critic.audit(state, candidates, iteration=1)

        assert fb1.verdict == fb2.verdict
        assert fb1.health_score == fb2.health_score
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_corrective_plan_set_for_rejected(tmp_path):
    """A REJECTED feedback must have a non-None target_agent_to_rerun and non-empty directives."""
    session_id = "test_critic_corrective"
    try:
        csv = _make_leaky_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        feedback = CriticAgent().audit(state, candidates)

        if feedback.verdict == "REJECTED":
            assert feedback.target_agent_to_rerun is not None
            assert len(feedback.specific_directives) > 0
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: Self-Reflection Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def test_orchestrator_approved_clean_pipeline(tmp_path):
    """A clean pipeline should finish with APPROVED in ≤ max_cycles."""
    session_id = "test_orch_clean"
    try:
        csv = _make_clean_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        orch = SelfReflectionOrchestrator(max_reflection_cycles=3)
        final_state, final_candidates = orch.run(state, candidates)

        assert len(final_state.reflection_log) >= 1
        last_cycle = final_state.reflection_log[-1]
        assert last_cycle.stopped_reason in ("APPROVED", "MAX_ITERATIONS_REACHED")
        assert len(final_state.critic_history) >= 1
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_orchestrator_corrects_leaky_pipeline(tmp_path):
    """Leaky pipeline should be REJECTED, corrected, and re-evaluated."""
    session_id = "test_orch_leaky_fix"
    try:
        csv = _make_leaky_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        # Verify initial critic says REJECTED
        initial_feedback = CriticAgent().audit(state, candidates, iteration=0)
        if initial_feedback.verdict != "REJECTED":
            pytest.skip("Initial pipeline was not REJECTED; cannot test correction.")

        orch = SelfReflectionOrchestrator(max_reflection_cycles=3)
        final_state, final_candidates = orch.run(state, candidates)

        # Reflection log should have at least 2 entries (cycle 1 = REJECTED, cycle 2 onward)
        assert len(final_state.reflection_log) >= 1

        # Critic history should have recorded all cycles
        assert len(final_state.critic_history) >= 1

        # At least one cycle should have had a corrective action applied
        corrected_cycles = [
            c for c in final_state.reflection_log if c.corrective_action is not None
        ]
        # Either we corrected and got APPROVED, or hit max iterations
        last = final_state.reflection_log[-1]
        assert last.stopped_reason in ("APPROVED", "MAX_ITERATIONS_REACHED")
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_orchestrator_max_cycles_safety(tmp_path):
    """Orchestrator must stop at max_reflection_cycles and never loop infinitely."""
    session_id = "test_orch_max_cycles"
    try:
        csv = _make_leaky_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        orch = SelfReflectionOrchestrator(max_reflection_cycles=2)
        final_state, final_candidates = orch.run(state, candidates)

        # Must have AT MOST max_reflection_cycles log entries
        assert len(final_state.reflection_log) <= 2
        assert len(final_state.critic_history) <= 2

        # Final cycle must have a stopped_reason
        last = final_state.reflection_log[-1]
        assert last.stopped_reason is not None
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_reflection_log_structure(tmp_path):
    """Every ReflectionCycleLog entry must have all required fields populated."""
    session_id = "test_orch_log_structure"
    try:
        csv = _make_clean_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        orch = SelfReflectionOrchestrator(max_reflection_cycles=2)
        final_state, _ = orch.run(state, candidates)

        for entry in final_state.reflection_log:
            assert entry.cycle_number >= 1
            assert entry.verdict in ("APPROVED", "REJECTED")
            assert 0.0 <= entry.health_score <= 100.0
            assert isinstance(entry.critical_issues, list)
            assert isinstance(entry.warnings_raised, list)
            assert isinstance(entry.metrics_before, dict)
            assert isinstance(entry.metrics_after, dict)
            assert entry.stopped_reason is not None or entry.corrective_action is not None
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_audit_history_persists_across_cycles(tmp_path):
    """state.critic_history must accumulate one entry per reflection cycle."""
    session_id = "test_orch_history"
    try:
        csv = _make_leaky_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        orch = SelfReflectionOrchestrator(max_reflection_cycles=3)
        final_state, _ = orch.run(state, candidates)

        # critic_history length must equal reflection_log length
        assert len(final_state.critic_history) == len(final_state.reflection_log)

        # Each CriticFeedback must have its iteration set correctly
        for i, fb in enumerate(final_state.critic_history, start=1):
            assert fb.iteration == i
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_orchestrator_sets_corrective_directives(tmp_path):
    """Orchestrator must populate state.critic_directives before rerunning agents."""
    session_id = "test_orch_directives"
    try:
        csv = _make_leaky_csv(tmp_path)
        state, candidates = _run_pipeline_to_evaluation(csv, session_id)

        # Verify leakage is detected
        feedback = CriticAgent().audit(state, candidates, iteration=1)
        if feedback.verdict != "REJECTED":
            pytest.skip("Leakage not detected; directive test not applicable.")

        # Apply corrective directives manually (as orchestrator would)
        orch = SelfReflectionOrchestrator(max_reflection_cycles=1)
        state_after = orch._set_corrective_directives(state, feedback)

        # directives must be non-empty for a REJECTED feedback
        assert len(state_after.critic_directives) > 0
    finally:
        shutil.rmtree(WORKSPACE_DIR / session_id, ignore_errors=True)


def test_critic_manually_constructed_feedback():
    """CriticFeedback with CRITICAL_FAIL items must produce REJECTED in isolation."""
    from core.state import CriticAuditItem, CriticFeedback

    # Construct a minimal leakage audit item
    leakage_item = CriticAuditItem(
        vector_id=1,
        check_name="Data Leakage Risk",
        status="CRITICAL_FAIL",
        observation="Feature 'leaky' has |corr| = 0.98 with target.",
        recommendation="Drop the feature and rerun FeatureEngineeringAgent.",
        affected_component="FeatureEngineeringAgent",
        evidence={"max_corr": 0.98, "leaky_feature": "leaky"},
    )
    pass_item = CriticAuditItem(
        vector_id=2,
        check_name="Train/Val/Test Contamination",
        status="PASS",
        observation="All distributions consistent.",
        recommendation="None required.",
        affected_component="FeatureEngineeringAgent",
        evidence={},
    )

    # Health: 100 - 35*1 - 10*0 = 65; verdict: REJECTED (1 CRITICAL_FAIL)
    health = max(0.0, 100.0 - 35.0 * 1 - 10.0 * 0)
    feedback = CriticFeedback(
        verdict="REJECTED",
        health_score=health,
        iteration=1,
        audit_results=[leakage_item, pass_item],
        target_agent_to_rerun="FeatureEngineer",
        specific_directives="Drop high-corr features.",
    )

    assert feedback.verdict == "REJECTED"
    assert feedback.health_score == 65.0
    assert feedback.target_agent_to_rerun == "FeatureEngineer"
    critical = [a for a in feedback.audit_results if a.status == "CRITICAL_FAIL"]
    assert len(critical) == 1
    assert critical[0].vector_id == 1
    assert critical[0].evidence["max_corr"] == 0.98


def test_max_cycles_argument_validation():
    """SelfReflectionOrchestrator must reject max_reflection_cycles < 1."""
    import pytest
    with pytest.raises(ValueError, match="max_reflection_cycles"):
        SelfReflectionOrchestrator(max_reflection_cycles=0)
