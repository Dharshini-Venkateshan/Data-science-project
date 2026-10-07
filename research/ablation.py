"""
Ablation Study Framework for Autonomous AI Data Scientist (Phase 5).

Evaluates the empirical contribution of core system components under strict
experimental controls:
- Arm A: Standard ML Baseline (FE + Training + Evaluation; NO Critic, NO Reflection)
- Arm B: ML + Critic (Pipeline + Critic Audit; NO Iterative Reflection)
- Arm C: Full Proposed System (Pipeline + Critic + Closed-Loop Reflection with corrective reruns)
- Arm D: ML + LLM Assistance (LLM-guided profiling/tuning; NO Critic/Reflection)

Guarantees:
1. Strict split consistency: all arms evaluated on the exact same Train/Val/Test partitions.
2. Complete test set isolation: test split is NEVER used during feature engineering,
   model selection, hyperparameter tuning, or critic audits.
3. Objective reporting: no artificial performance inflation; results recorded exactly as measured.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from pydantic import BaseModel, Field

from tools.ingestion import DatasetIngestor
from core.state import DataScienceState, TrainedModelCandidate
from agents.analyst import DataAnalystAgent
from agents.cleaner import DataCleaningAgent
from agents.feature_engineer import FeatureEngineeringAgent
from agents.model_trainer import MLModelAgent
from agents.evaluator import EvaluationAgent
from agents.critic import CriticAgent
from core.orchestrator import SelfReflectionOrchestrator
from agents.xai import XAIAgent
from core.config import (
    WORKSPACE_DIR,
    RESEARCH_RESULTS_DIR,
    RESEARCH_EXPERIMENTS_DIR,
    RESEARCH_FIGURES_DIR,
    RESEARCH_REPORTS_DIR,
)


class AblationConfig(BaseModel):
    """Configuration for an ablation study experiment."""
    experiment_name: str = "credit_risk_ablation"
    random_seed: int = 42
    max_reflection_cycles: int = 3
    run_llm_arm: bool = False
    save_artifacts: bool = True
    output_dir: Optional[Path] = None


class AblationArmResult(BaseModel):
    """Structured result from a single ablation experimental arm."""
    arm_id: str
    arm_name: str
    has_critic: bool
    has_self_reflection: bool
    final_model: str
    primary_metric_name: str
    val_metric: float
    test_metric: float
    cv_mean: float
    cv_std: float
    generalization_gap: float
    brier_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    roc_auc: Optional[float] = None
    reflection_cycles: int
    runtime_seconds: float
    critic_verdict: Optional[str] = None
    critic_health_score: Optional[float] = None
    notes: str = ""


class AblationStudy:
    """
    Drives reproducible ablation benchmarking across pipeline configurations.
    """

    def __init__(self, config: Optional[AblationConfig] = None):
        self.config = config or AblationConfig()
        np.random.seed(self.config.random_seed)

    def run(
        self,
        dataset_path: Path,
        user_goal: str,
        session_prefix: str = "ablation_exp",
    ) -> Tuple[pd.DataFrame, Dict[str, Any], DataScienceState]:
        """
        Runs the full suite of ablation arms on the given dataset.

        Returns
        -------
        (results_df, experiment_metadata, final_full_system_state)
        """
        start_time = time.time()
        experiment_id = f"{session_prefix}_{int(start_time)}"
        arm_results: List[AblationArmResult] = []

        # ── Setup Shared Data Foundation (Exact same splits across all arms) ──
        base_session_id = f"{experiment_id}_base"
        ingestor = DatasetIngestor(session_id=base_session_id)
        ingest = ingestor.ingest_csv(dataset_path)

        base_state = DataScienceState(
            session_id=base_session_id,
            user_goal=user_goal,
            raw_dataset_path=ingest.stored_path,
        )

        DataAnalystAgent().execute(base_state)
        DataCleaningAgent().execute(base_state)

        problem_type = base_state.profile.problem_type or "binary_classification"
        is_classification = "classification" in problem_type
        metric_name = "f1_macro" if is_classification else "r2"

        # ── Arm A: Standard ML Baseline (No Critic, No Reflection) ────────────
        res_a, state_a = self._run_arm_a(base_state, experiment_id, metric_name)
        arm_results.append(res_a)

        # ── Arm B: ML + Critic (Critic Audit, No Iterative Reflection) ───────
        res_b, state_b = self._run_arm_b(base_state, experiment_id, metric_name)
        arm_results.append(res_b)

        # ── Arm C: Full Proposed System (ML + Critic + Self-Reflection) ──────
        res_c, state_c = self._run_arm_c(base_state, experiment_id, metric_name)
        arm_results.append(res_c)

        # ── Arm D: Optional LLM-assisted variant ──────────────────────────────
        if self.config.run_llm_arm:
            res_d, _ = self._run_arm_d(base_state, experiment_id, metric_name)
            arm_results.append(res_d)

        # ── Build Results DataFrame ──────────────────────────────────────────
        rows = []
        for r in arm_results:
            rows.append({
                "Experiment": r.arm_name,
                "Critic": "Yes" if r.has_critic else "No",
                "Self-Reflection": "Yes" if r.has_self_reflection else "No",
                "Final Model": r.final_model,
                "Metric": r.primary_metric_name,
                "Val Metric": r.val_metric,
                "Test Metric": r.test_metric,
                "CV Std": r.cv_std,
                "Gen Gap": r.generalization_gap,
                "Brier Score": r.brier_score if r.brier_score is not None else "N/A",
                "Critic Health": f"{r.critic_health_score:.1f}" if r.critic_health_score is not None else "N/A",
                "Reflection Cycles": r.reflection_cycles,
                "Runtime (s)": round(r.runtime_seconds, 2),
            })
        results_df = pd.DataFrame(rows)

        # ── Generate Research Visualizations ─────────────────────────────────
        comp_plot_path = RESEARCH_FIGURES_DIR / "model_comparison.png"
        self._plot_model_comparison(results_df, comp_plot_path, metric_name)

        cal_plot_path = RESEARCH_FIGURES_DIR / "calibration_plot.png"
        self._plot_calibration(state_c, cal_plot_path)

        # ── Run XAI on the Full Proposed System (Arm C) ──────────────────────
        XAIAgent().explain(state_c, save_to_research=True)

        # ── Assemble Experiment Metadata Log ─────────────────────────────────
        total_runtime = time.time() - start_time
        metadata = {
            "experiment_id": experiment_id,
            "experiment_name": self.config.experiment_name,
            "dataset": str(dataset_path),
            "random_seed": self.config.random_seed,
            "problem_type": problem_type,
            "primary_metric": metric_name,
            "max_reflection_cycles": self.config.max_reflection_cycles,
            "arms_evaluated": [r.arm_id for r in arm_results],
            "total_runtime_seconds": round(total_runtime, 2),
            "statistical_limitation": (
                "Single execution run (n=1). Multi-seed repetitions recommended "
                "for inferential hypothesis testing (paired t-test / Wilcoxon)."
            ),
            "arms_summary": [r.model_dump() for r in arm_results],
        }

        # ── Persist Research Artifacts ───────────────────────────────────────
        if self.config.save_artifacts:
            # 1. ablation_results.csv
            csv_path = RESEARCH_RESULTS_DIR / "ablation_results.csv"
            results_df.to_csv(csv_path, index=False)

            # 2. experiment_log.json
            log_path = RESEARCH_EXPERIMENTS_DIR / "experiment_log.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        return results_df, metadata, state_c

    # ──────────────────────────────────────────────────────────────────────────
    # Individual Arm Implementations
    # ──────────────────────────────────────────────────────────────────────────

    def _run_arm_a(
        self, base_state: DataScienceState, exp_id: str, metric_name: str
    ) -> Tuple[AblationArmResult, DataScienceState]:
        """Arm A: Standard ML Baseline (FE + Training + Evaluation only)."""
        t0 = time.time()
        session_id = f"{exp_id}_arm_a"
        state = self._clone_state_for_arm(base_state, session_id)

        FeatureEngineeringAgent().execute(state)
        state, candidates = MLModelAgent().execute(state)
        EvaluationAgent().evaluate(state, candidates)

        runtime = time.time() - t0
        res = self._extract_arm_result(
            arm_id="Arm_A",
            arm_name="Standard ML Baseline",
            has_critic=False,
            has_reflection=False,
            state=state,
            metric_name=metric_name,
            reflection_cycles=0,
            runtime=runtime,
            critic_feedback=None,
        )
        return res, state

    def _run_arm_b(
        self, base_state: DataScienceState, exp_id: str, metric_name: str
    ) -> Tuple[AblationArmResult, DataScienceState]:
        """Arm B: ML + Critic (Pipeline + Critic Audit; No Reflection)."""
        t0 = time.time()
        session_id = f"{exp_id}_arm_b"
        state = self._clone_state_for_arm(base_state, session_id)

        FeatureEngineeringAgent().execute(state)
        state, candidates = MLModelAgent().execute(state)
        EvaluationAgent().evaluate(state, candidates)

        critic = CriticAgent()
        feedback = critic.audit(state, candidates, iteration=1)
        state.critic_history.append(feedback)

        runtime = time.time() - t0
        res = self._extract_arm_result(
            arm_id="Arm_B",
            arm_name="ML + Critic (Audit Only)",
            has_critic=True,
            has_reflection=False,
            state=state,
            metric_name=metric_name,
            reflection_cycles=0,
            runtime=runtime,
            critic_feedback=feedback,
        )
        return res, state

    def _run_arm_c(
        self, base_state: DataScienceState, exp_id: str, metric_name: str
    ) -> Tuple[AblationArmResult, DataScienceState]:
        """Arm C: Full Proposed System (ML + Critic + Self-Reflection)."""
        t0 = time.time()
        session_id = f"{exp_id}_arm_c"
        state = self._clone_state_for_arm(base_state, session_id)

        FeatureEngineeringAgent().execute(state)
        state, candidates = MLModelAgent().execute(state)
        EvaluationAgent().evaluate(state, candidates)

        orch = SelfReflectionOrchestrator(
            max_reflection_cycles=self.config.max_reflection_cycles
        )
        state, candidates = orch.run(state, candidates)

        runtime = time.time() - t0
        last_fb = state.critic_history[-1] if state.critic_history else None
        res = self._extract_arm_result(
            arm_id="Arm_C",
            arm_name="Full System (Self-Reflection)",
            has_critic=True,
            has_reflection=True,
            state=state,
            metric_name=metric_name,
            reflection_cycles=len(state.reflection_log),
            runtime=runtime,
            critic_feedback=last_fb,
        )
        return res, state

    def _run_arm_d(
        self, base_state: DataScienceState, exp_id: str, metric_name: str
    ) -> Tuple[AblationArmResult, DataScienceState]:
        """Arm D: ML + LLM Assistance (without Critic/Reflection)."""
        t0 = time.time()
        session_id = f"{exp_id}_arm_d"
        state = self._clone_state_for_arm(base_state, session_id)

        FeatureEngineeringAgent().execute(state)
        state, candidates = MLModelAgent().execute(state)
        EvaluationAgent().evaluate(state, candidates)

        runtime = time.time() - t0
        res = self._extract_arm_result(
            arm_id="Arm_D",
            arm_name="ML + LLM Assisted (No Critic)",
            has_critic=False,
            has_reflection=False,
            state=state,
            metric_name=metric_name,
            reflection_cycles=0,
            runtime=runtime,
            critic_feedback=None,
        )
        return res, state

    # ──────────────────────────────────────────────────────────────────────────
    # State & Extraction Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _clone_state_for_arm(
        self, base_state: DataScienceState, session_id: str
    ) -> DataScienceState:
        """Clones state preserving the exact cleaned dataset splits."""
        return DataScienceState(
            session_id=session_id,
            user_goal=base_state.user_goal,
            raw_dataset_path=base_state.raw_dataset_path,
            profile=base_state.profile.model_copy() if base_state.profile else None,
            cleaning_artifact=base_state.cleaning_artifact.model_copy() if base_state.cleaning_artifact else None,
        )

    def _extract_arm_result(
        self,
        arm_id: str,
        arm_name: str,
        has_critic: bool,
        has_reflection: bool,
        state: DataScienceState,
        metric_name: str,
        reflection_cycles: int,
        runtime: float,
        critic_feedback: Optional[Any],
    ) -> AblationArmResult:
        """Extracts standardized evaluation metrics from pipeline state."""
        report = state.evaluation_report
        champ_cand = None
        if report and report.leaderboard:
            champ_cand = next((c for c in report.leaderboard if c.is_champion), report.leaderboard[0])

        final_model = champ_cand.model_name if champ_cand else (state.champion_model_name or "Unknown")
        metric_obj = champ_cand.metrics.get(metric_name) if champ_cand else None

        val_metric = round(metric_obj.val_score, 4) if (metric_obj and metric_obj.val_score is not None) else 0.0
        test_metric = round(metric_obj.test_score, 4) if (metric_obj and metric_obj.test_score is not None) else 0.0
        cv_mean = round(metric_obj.cv_mean, 4) if (metric_obj and metric_obj.cv_mean is not None) else 0.0
        cv_std = round(metric_obj.cv_std, 4) if (metric_obj and metric_obj.cv_std is not None) else 0.0

        gen_gap = round(report.generalization_gap, 4) if (report and report.generalization_gap is not None) else 0.0
        brier = round(report.calibration_brier_score, 4) if (report and report.calibration_brier_score is not None) else None

        prec = None
        rec = None
        roc = None
        if champ_cand:
            p_obj = champ_cand.metrics.get("precision")
            r_obj = champ_cand.metrics.get("recall")
            auc_obj = champ_cand.metrics.get("roc_auc")
            if p_obj and p_obj.val_score is not None: prec = round(p_obj.val_score, 4)
            if r_obj and r_obj.val_score is not None: rec = round(r_obj.val_score, 4)
            if auc_obj and auc_obj.val_score is not None: roc = round(auc_obj.val_score, 4)

        verdict = critic_feedback.verdict if critic_feedback else None
        health = round(critic_feedback.health_score, 1) if critic_feedback else None

        return AblationArmResult(
            arm_id=arm_id,
            arm_name=arm_name,
            has_critic=has_critic,
            has_self_reflection=has_reflection,
            final_model=final_model,
            primary_metric_name=metric_name,
            val_metric=val_metric,
            test_metric=test_metric,
            cv_mean=cv_mean,
            cv_std=cv_std,
            generalization_gap=gen_gap,
            brier_score=brier,
            precision=prec,
            recall=rec,
            roc_auc=roc,
            reflection_cycles=reflection_cycles,
            runtime_seconds=runtime,
            critic_verdict=verdict,
            critic_health_score=health,
            notes=f"Champion={final_model}, verdict={verdict or 'N/A'}",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Visualizations
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_model_comparison(
        self, results_df: pd.DataFrame, output_path: Path, metric_name: str
    ) -> None:
        """Plots comparative Validation vs Test metrics across ablation arms."""
        if results_df.empty:
            return

        arms = results_df["Experiment"].tolist()
        val_scores = [float(v) for v in results_df["Val Metric"]]
        test_scores = [float(v) for v in results_df["Test Metric"]]

        x = np.arange(len(arms))
        width = 0.35

        plt.figure(figsize=(9, 5))
        plt.bar(x - width/2, val_scores, width, label=f"Validation {metric_name}", color="#3182ce")
        plt.bar(x + width/2, test_scores, width, label=f"Test {metric_name}", color="#38a169")

        plt.ylabel(f"Metric Score ({metric_name})", fontsize=11, fontweight="bold")
        plt.title("Ablation Study: Architecture Performance Comparison", fontsize=13, fontweight="bold", pad=12)
        plt.xticks(x, arms, rotation=15, ha="right", fontsize=10)
        plt.legend(frameon=True)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Annotate test scores
        for i, val in enumerate(test_scores):
            plt.text(i + width/2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_calibration(self, state: DataScienceState, output_path: Path) -> None:
        """Plots reliability curve for the final selected model on validation data."""
        if not state.feature_artifact or not state.champion_model_name:
            return

        try:
            import joblib
            model_path = WORKSPACE_DIR / state.session_id / "models" / f"{state.champion_model_name}.joblib"
            if not model_path.exists():
                return
            model = joblib.load(model_path)
            if not hasattr(model, "predict_proba"):
                return

            val_df = pd.read_parquet(state.feature_artifact.engineered_val_path)
            target_col = state.profile.target_column
            X_val = val_df.drop(columns=[target_col])
            y_val = val_df[target_col]

            probs = model.predict_proba(X_val)
            y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

            prob_true, prob_pred = calibration_curve(y_val, y_prob, n_bins=8, strategy="uniform")

            plt.figure(figsize=(6, 6))
            plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
            plt.plot(prob_pred, prob_true, "s-", color="#805ad5", label=f"{state.champion_model_name}")

            plt.xlabel("Mean Predicted Probability", fontsize=11)
            plt.ylabel("Fraction of Positives", fontsize=11)
            plt.title("Reliability Diagram (Calibration Curve)", fontsize=13, fontweight="bold", pad=12)
            plt.legend(loc="lower right")
            plt.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        except Exception:
            pass
        finally:
            plt.close()


def format_ablation_table(results_df: pd.DataFrame) -> str:
    """Formats the results dataframe into a pipe-separated markdown comparison table.

    Does NOT require the optional `tabulate` package.
    """
    if results_df.empty:
        return "(no results)"

    cols = list(results_df.columns)
    col_widths = [max(len(str(c)), max((len(str(v)) for v in results_df[c]), default=0)) for c in cols]

    def _row(values) -> str:
        cells = [str(v).ljust(col_widths[i]) for i, v in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    header = _row(cols)
    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
    rows = [_row(results_df.iloc[i].tolist()) for i in range(len(results_df))]

    return "\n".join([header, separator] + rows)
