"""
Self-Reflection Orchestrator for Autonomous AI Data Scientist (Phase 4).

Implements a closed-loop, multi-cycle reflection engine that:

  Pipeline → Evaluation → Critic Audit
      ↓ APPROVED → COMPLETED
      ↓ REJECTED
          → identify failed component
          → set corrective directives
          → rerun the responsible agent
          → re-evaluate
          → critic audits again
          (repeat up to max_reflection_cycles)

Every cycle is fully recorded in state.reflection_log as a ReflectionCycleLog
entry so the entire decision trail is scientifically auditable.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from agents.critic import CriticAgent
from agents.feature_engineer import FeatureEngineeringAgent
from agents.model_trainer import MLModelAgent
from agents.evaluator import EvaluationAgent
from core.state import (
    DataScienceState,
    CriticFeedback,
    ReflectionCycleLog,
    TrainedModelCandidate,
)


class SelfReflectionOrchestrator:
    """
    Drives the closed-loop self-reflection cycle.

    Parameters
    ----------
    max_reflection_cycles : int
        Hard ceiling on how many REJECTED → correct → re-evaluate cycles can
        run.  Prevents infinite loops.  After this many cycles the last state
        is returned regardless of verdict.
    """

    def __init__(self, max_reflection_cycles: int = 3):
        if max_reflection_cycles < 1:
            raise ValueError("max_reflection_cycles must be at least 1.")
        self.max_reflection_cycles = max_reflection_cycles
        self._critic = CriticAgent()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        state: DataScienceState,
        candidates: List[TrainedModelCandidate],
    ) -> Tuple[DataScienceState, List[TrainedModelCandidate]]:
        """
        Runs the self-reflection loop starting from an already-evaluated
        pipeline state.

        Preconditions
        -------------
        - state.feature_artifact must be populated (FeatureEngineeringAgent done)
        - state.evaluation_report must be populated (EvaluationAgent done)
        - candidates must be the list returned by MLModelAgent.execute()

        Returns
        -------
        (final_state, final_candidates)
            final_state has reflection_log fully populated.
        """
        for cycle in range(1, self.max_reflection_cycles + 1):
            state.status = "CRITIC_REVIEW"
            state.current_iteration = cycle

            # ── Collect pre-cycle metrics ──────────────────────────────────
            metrics_before = self._champion_metrics_snapshot(state, candidates)
            champion_before = state.champion_model_name

            # ── Critic Audit ───────────────────────────────────────────────
            feedback: CriticFeedback = self._critic.audit(state, candidates, iteration=cycle)
            state.critic_history.append(feedback)

            # ── Build cycle log entry ──────────────────────────────────────
            log_entry = ReflectionCycleLog(
                cycle_number=cycle,
                verdict=feedback.verdict,
                health_score=feedback.health_score,
                critical_issues=[
                    a.check_name
                    for a in feedback.audit_results
                    if a.status == "CRITICAL_FAIL"
                ],
                warnings_raised=[
                    a.check_name
                    for a in feedback.audit_results
                    if a.status == "WARNING"
                ],
                champion_before=champion_before,
                metrics_before=metrics_before,
                directive_summary=feedback.specific_directives,
            )

            # ── APPROVED path ──────────────────────────────────────────────
            if feedback.verdict == "APPROVED":
                log_entry.stopped_reason = "APPROVED"
                state.reflection_log.append(log_entry)
                state.status = "COMPLETED"
                state.log(
                    f"SelfReflectionOrchestrator: APPROVED at cycle {cycle}. "
                    f"Health score = {feedback.health_score}/100."
                )
                break

            # ── Safety net: max cycles reached ────────────────────────────
            if cycle >= self.max_reflection_cycles:
                log_entry.stopped_reason = "MAX_ITERATIONS_REACHED"
                state.reflection_log.append(log_entry)
                state.status = "COMPLETED"
                state.log(
                    f"SelfReflectionOrchestrator: max_reflection_cycles "
                    f"({self.max_reflection_cycles}) reached. "
                    f"Last verdict = REJECTED (health={feedback.health_score}/100). "
                    f"Stopping with current champion."
                )
                break

            # ── REJECTED path: apply correction and rerun ──────────────────
            log_entry.corrective_action = feedback.specific_directives
            log_entry.agent_rerun       = feedback.target_agent_to_rerun

            state.log(
                f"SelfReflectionOrchestrator cycle {cycle}: REJECTED. "
                f"Target={feedback.target_agent_to_rerun}. "
                f"Directive: {feedback.specific_directives[:120]}..."
            )

            # Set machine-readable directives before rerunning agents
            state = self._set_corrective_directives(state, feedback)
            state.status = "RE_PLANNING"

            # Rerun the responsible agent chain
            state, candidates = self._rerun_agent_chain(
                state, feedback.target_agent_to_rerun, candidates
            )

            # Capture post-correction metrics for the log
            metrics_after  = self._champion_metrics_snapshot(state, candidates)
            champion_after = state.champion_model_name
            log_entry.metrics_after  = metrics_after
            log_entry.champion_after = champion_after
            state.reflection_log.append(log_entry)

        return state, candidates

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _set_corrective_directives(
        self, state: DataScienceState, feedback: CriticFeedback
    ) -> DataScienceState:
        """
        Translates CriticFeedback into a machine-readable directives dict stored
        in state.critic_directives.  The downstream agents read this dict to
        alter their behaviour.
        """
        directives: Dict[str, Any] = {}

        for item in feedback.audit_results:
            if item.status not in ("CRITICAL_FAIL", "WARNING"):
                continue

            vid = item.vector_id

            if vid == 1:   # Data Leakage
                directives["drop_high_corr_features"] = True
                directives["corr_threshold"] = 0.85

            elif vid == 2:  # Contamination
                directives["strict_train_only"] = True

            elif vid == 3:  # Overfitting
                directives["stronger_regularization"] = True

            elif vid == 4:  # Class Imbalance
                directives["enhanced_class_balance"] = True

            elif vid == 5:  # High Val Perf (also implies leakage)
                directives["drop_high_corr_features"] = True
                directives["corr_threshold"] = 0.85

            elif vid == 6:  # Unstable CV
                directives["increase_cv_folds"] = True
                directives["n_cv_splits"]        = 7

            elif vid == 7:  # Poor Calibration
                directives["apply_calibration"] = True

            elif vid == 8:  # Preprocessing
                directives["strict_train_only"] = True

        state.critic_directives = directives
        state.log(
            f"SelfReflectionOrchestrator: applied directives: {list(directives.keys())}"
        )
        return state

    def _rerun_agent_chain(
        self,
        state: DataScienceState,
        target: Optional[str],
        current_candidates: List[TrainedModelCandidate],
    ) -> Tuple[DataScienceState, List[TrainedModelCandidate]]:
        """
        Re-executes the pipeline from the specified agent and all downstream
        agents, so that evaluation reflects the corrected artefacts.

        Chain dependencies
        ──────────────────
        FeatureEngineer → ModelTrainer → Evaluator
        ModelTrainer    → Evaluator
        Evaluator       (standalone)
        """
        candidates = current_candidates  # default (may be replaced below)

        if target == "FeatureEngineer":
            state = FeatureEngineeringAgent().execute(state)
            state, candidates = MLModelAgent().execute(state)
            state = EvaluationAgent().evaluate(state, candidates)

        elif target == "ModelTrainer":
            state, candidates = MLModelAgent().execute(state)
            state = EvaluationAgent().evaluate(state, candidates)

        elif target == "Evaluator":
            state = EvaluationAgent().evaluate(state, candidates)

        else:
            # Unknown target – fall back to full retrain from feature engineering
            state.log(
                f"SelfReflectionOrchestrator: unknown target '{target}'. "
                "Falling back to full ModelTrainer + Evaluator rerun."
            )
            state, candidates = MLModelAgent().execute(state)
            state = EvaluationAgent().evaluate(state, candidates)

        return state, candidates

    # ──────────────────────────────────────────────────────────────────────────
    # Metrics snapshot helper
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _champion_metrics_snapshot(
        state: DataScienceState,
        candidates: List[TrainedModelCandidate],
    ) -> Dict[str, float]:
        """Extracts key metrics of the current champion for before/after comparison."""
        report = state.evaluation_report
        if not report or not report.leaderboard:
            return {}

        champion = next((c for c in report.leaderboard if c.is_champion), None)
        if not champion:
            return {}

        primary = "f1_macro" if "classification" in (state.profile.problem_type or "") else "r2"
        m = champion.metrics.get(primary)
        if not m:
            return {}

        snapshot: Dict[str, float] = {}
        if m.val_score  is not None: snapshot["val_score"]   = round(m.val_score,  4)
        if m.test_score is not None: snapshot["test_score"]  = round(m.test_score, 4)
        if m.cv_mean    is not None: snapshot["cv_mean"]     = round(m.cv_mean,    4)
        if m.cv_std     is not None: snapshot["cv_std"]      = round(m.cv_std,     4)
        if m.train_score is not None: snapshot["train_score"] = round(m.train_score, 4)

        if report.generalization_gap is not None:
            snapshot["generalization_gap"] = round(report.generalization_gap, 4)
        if report.calibration_brier_score is not None:
            snapshot["brier_score"] = round(report.calibration_brier_score, 4)

        return snapshot
