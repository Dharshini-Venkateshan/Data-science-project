"""
Evaluation Engine Agent for Autonomous AI Data Scientist.
Ranks candidates via validation performance and CV stability, preserves test set isolation
until final champion evaluation, and computes comprehensive multi-metric diagnostics.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from agents.base import BaseAgent
from core.state import DataScienceState, EvaluationReport, TrainedModelCandidate, ModelEvaluationMetric
from core.llm_client import BaseLLMClient


class EvaluationAgent(BaseAgent):
    """
    Evaluates candidate tournament models and generates structured diagnostic leaderboards.
    Guarantees test partition remains untouched until champion selection is finalized.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        super().__init__(name="EvaluationAgent", llm_client=llm_client)

    def _compute_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: Any,
        n_bootstraps: int = 100
    ) -> Tuple[float, float]:
        """Calculates 95% confidence intervals via empirical bootstrapping."""
        scores = []
        n_samples = len(y_true)
        rng = np.random.RandomState(42)

        for _ in range(n_bootstraps):
            indices = rng.randint(0, n_samples, n_samples)
            if len(np.unique(y_true[indices])) < 2 and metric_func in [roc_auc_score]:
                continue
            try:
                score = metric_func(y_true[indices], y_pred[indices])
                scores.append(score)
            except Exception:
                continue

        if not scores:
            return 0.0, 0.0

        lower = float(np.percentile(scores, 2.5))
        upper = float(np.percentile(scores, 97.5))
        return round(lower, 4), round(upper, 4)

    def evaluate(
        self,
        state: DataScienceState,
        candidate_records: List[TrainedModelCandidate]
    ) -> DataScienceState:
        """
        Ranks candidates based on validation scores and evaluates the champion on the untouched test split.
        """
        if not candidate_records:
            raise ValueError("EvaluationAgent requires at least one candidate model record.")

        target_col = state.profile.target_column
        problem_type = state.profile.problem_type
        is_classification = "classification" in problem_type
        primary_metric_name = "f1_macro" if is_classification else "r2"

        # 1. Rank Candidate Models using Validation Metric & CV Stability
        # Composite selection score: Val_score - (0.25 * CV_std) to favor robust models
        def selection_score(c: TrainedModelCandidate) -> float:
            m = c.metrics.get(primary_metric_name)
            if not m:
                return -999.0
            val = m.val_score or 0.0
            std = m.cv_std or 0.0
            return val - (0.25 * std)

        ranked_candidates = sorted(candidate_records, key=selection_score, reverse=True)
        champion = ranked_candidates[0]
        champion.is_champion = True

        # 2. FINAL UNTOUCHED TEST SET EVALUATION (Only for the selected Champion!)
        test_df = pd.read_parquet(state.feature_artifact.engineered_test_path)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].values

        champion_model = joblib.load(champion.artifact_path)
        test_preds = champion_model.predict(X_test)

        # 3. Compute Comprehensive Test Metrics
        conf_matrix: Optional[List[List[int]]] = None
        class_report: Optional[Dict[str, Any]] = None
        residual_diag: Optional[Dict[str, Any]] = None
        brier_score: Optional[float] = None

        if is_classification:
            test_f1_macro = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
            test_acc = float(accuracy_score(y_test, test_preds))
            conf_matrix = confusion_matrix(y_test, test_preds).tolist()
            class_report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)

            # Bootstrapping 95% CI
            ci_lower, ci_upper = self._compute_bootstrap_ci(
                y_test, test_preds, lambda y, p: f1_score(y, p, average="macro", zero_division=0)
            )

            # Brier Score if probabilities are available
            if hasattr(champion_model, "predict_proba"):
                try:
                    test_probs = champion_model.predict_proba(X_test)
                    if test_probs.shape[1] == 2:
                        brier_score = float(brier_score_loss(y_test, test_probs[:, 1]))
                except Exception:
                    pass

            primary_test_score = test_f1_macro

        else:  # Regression
            test_r2 = float(r2_score(y_test, test_preds))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
            test_mae = float(mean_absolute_error(y_test, test_preds))

            # Residual analysis
            residuals = y_test - test_preds
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000]) if len(residuals) >= 3 else (0.0, 1.0)
            residual_diag = {
                "residual_mean": round(float(np.mean(residuals)), 4),
                "residual_std": round(float(np.std(residuals)), 4),
                "max_error": round(float(np.max(np.abs(residuals))), 4),
                "normality_p_value": round(float(shapiro_p), 4),
                "rmse": round(test_rmse, 4),
                "mae": round(test_mae, 4)
            }

            ci_lower, ci_upper = self._compute_bootstrap_ci(y_test, test_preds, r2_score)
            primary_test_score = test_r2

        # 4. Attach Test Metrics to Champion
        champ_metric = champion.metrics[primary_metric_name]
        champ_metric.test_score = round(primary_test_score, 4)
        champ_metric.bootstrap_ci_lower = ci_lower
        champ_metric.bootstrap_ci_upper = ci_upper

        # 5. Generalization Gap Calculation
        gen_gap = None
        if champ_metric.val_score is not None and champ_metric.test_score is not None:
            gen_gap = round(abs(champ_metric.val_score - champ_metric.test_score), 4)

        # 6. Construct EvaluationReport Artifact
        report = EvaluationReport(
            leaderboard=ranked_candidates,
            champion_model_name=champion.model_name,
            generalization_gap=gen_gap,
            cv_variance=champ_metric.cv_std,
            calibration_brier_score=round(brier_score, 4) if brier_score is not None else None,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            residual_diagnostics=residual_diag
        )

        state.evaluation_report = report
        state.champion_model_name = champion.model_name
        state.status = "EVALUATING"
        state.log(
            f"EvaluationAgent: Selected champion '{champion.model_name}' "
            f"(Val {primary_metric_name}={champ_metric.val_score}, "
            f"Test {primary_metric_name}={champ_metric.test_score}, "
            f"CV Std={champ_metric.cv_std}). Leaderboard generated with {len(ranked_candidates)} models."
        )

        return state

    def execute(self, state: DataScienceState) -> DataScienceState:
        """Standard BaseAgent execution wrapper."""
        raise NotImplementedError(
            "EvaluationAgent.execute() requires candidate models from MLModelAgent. "
            "Use evaluate(state, candidate_records)."
        )
