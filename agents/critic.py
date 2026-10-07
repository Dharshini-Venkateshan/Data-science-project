"""
Critic Agent for Autonomous AI Data Scientist (Phase 4).

Adversarially audits the complete ML pipeline across 8 diagnostic vectors,
producing deterministic APPROVED / REJECTED verdicts and structured corrective
directives that the Self-Reflection Orchestrator can act on.

Audit Vectors
─────────────
V1  Data Leakage Risk              – feature–target Pearson correlation
V2  Train / Val / Test Contamination – scaled-feature distribution shift
V3  Overfitting & Generalization Gap – train–val gap, val–test gap
V4  Extreme Class Imbalance        – class_imbalance_ratio from profile
V5  Suspiciously High Val Perf.    – val_score anomaly detection
V6  Unstable Cross-Validation      – cv_std threshold
V7  Model Calibration              – Brier score assessment
V8  Preprocessing Consistency      – leakage guard and scaler sanity
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from agents.base import BaseAgent
from core.state import (
    DataScienceState,
    CriticAuditItem,
    CriticFeedback,
    TrainedModelCandidate,
)
from core.llm_client import BaseLLMClient


# ──────────────────────────────────────────────────────────────────────────────
# Threshold constants  (tuned so a well-behaved pipeline always gets APPROVED)
# ──────────────────────────────────────────────────────────────────────────────
_LEAKAGE_CORR_CRITICAL   = 0.92   # |feature–target| Pearson ≥ this → CRITICAL
_LEAKAGE_CORR_WARNING    = 0.80   # |feature–target| Pearson ≥ this → WARNING

_OVERFIT_GAP_CRITICAL    = 0.30   # train – val ≥ this            → CRITICAL
_OVERFIT_GAP_WARNING     = 0.15   # train – val ≥ this            → WARNING

_GEN_GAP_CRITICAL        = 0.25   # |val – test| ≥ this           → CRITICAL
_GEN_GAP_WARNING         = 0.12   # |val – test| ≥ this           → WARNING

_IMBALANCE_CRITICAL      = 8.0    # class_imbalance_ratio ≥ this  → CRITICAL
_IMBALANCE_WARNING       = 4.0    # class_imbalance_ratio ≥ this  → WARNING

_HIGH_VAL_CRITICAL       = 0.97   # val_score ≥ this              → CRITICAL
_HIGH_VAL_WARNING        = 0.90   # val_score ≥ this              → WARNING

_CV_STD_CRITICAL         = 0.18   # cv_std ≥ this                 → CRITICAL
_CV_STD_WARNING          = 0.10   # cv_std ≥ this                 → WARNING

_BRIER_CRITICAL          = 0.32   # Brier score ≥ this            → CRITICAL
_BRIER_WARNING           = 0.22   # Brier score ≥ this            → WARNING

_TRAIN_SCALED_MEAN_TOL   = 0.15   # |mean of scaled train col| > this → WARNING


class CriticAgent(BaseAgent):
    """
    Adversarial Critic Agent that audits the complete ML pipeline and produces
    structured, machine-readable CriticFeedback.

    Usage
    ─────
        feedback = CriticAgent().audit(state, candidates, iteration=1)
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        super().__init__(name="CriticAgent", llm_client=llm_client)

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def execute(self, state: DataScienceState) -> DataScienceState:
        """Standard BaseAgent hook – not used directly; call audit() instead."""
        raise NotImplementedError(
            "CriticAgent.execute() is not supported. "
            "Call audit(state, candidates, iteration) directly."
        )

    def audit(
        self,
        state: DataScienceState,
        candidates: List[TrainedModelCandidate],
        iteration: int = 1,
    ) -> CriticFeedback:
        """
        Runs all 8 diagnostic vectors and returns a CriticFeedback object.

        Parameters
        ----------
        state      : Current DataScienceState (must have feature_artifact,
                     evaluation_report, profile populated).
        candidates : List of TrainedModelCandidate from MLModelAgent.
        iteration  : Current reflection cycle number (1-indexed).

        Returns
        -------
        CriticFeedback with APPROVED / REJECTED verdict.
        """
        audit_results: List[CriticAuditItem] = []

        audit_results.append(self._v1_data_leakage(state))
        audit_results.append(self._v2_contamination(state))
        audit_results.append(self._v3_overfitting(state, candidates))
        audit_results.append(self._v4_class_imbalance(state))
        audit_results.append(self._v5_high_val_performance(state, candidates))
        audit_results.append(self._v6_cv_stability(state, candidates))
        audit_results.append(self._v7_calibration(state))
        audit_results.append(self._v8_preprocessing(state))

        n_critical = sum(1 for a in audit_results if a.status == "CRITICAL_FAIL")
        n_warning  = sum(1 for a in audit_results if a.status == "WARNING")

        health_score = max(0.0, 100.0 - (35.0 * n_critical) - (10.0 * n_warning))
        verdict: str = "REJECTED" if n_critical > 0 else "APPROVED"

        target_agent, directives = self._corrective_plan(audit_results, state)

        feedback = CriticFeedback(
            verdict=verdict,
            health_score=round(health_score, 1),
            iteration=iteration,
            audit_results=audit_results,
            target_agent_to_rerun=target_agent,
            specific_directives=directives,
        )

        state.log(
            f"CriticAgent (cycle {iteration}): verdict={verdict}, "
            f"health={health_score:.1f}/100, "
            f"critical={n_critical}, warnings={n_warning}."
        )
        return feedback

    # ──────────────────────────────────────────────────────────────────────────
    # Internal audit vector implementations
    # ──────────────────────────────────────────────────────────────────────────

    def _v1_data_leakage(self, state: DataScienceState) -> CriticAuditItem:
        """V1 – Feature–target Pearson correlation screen."""
        component = "FeatureEngineeringAgent"
        check = "Data Leakage Risk"

        if not (state.feature_artifact and state.feature_artifact.engineered_train_path):
            return self._make_item(1, check, "PASS", "No engineered data to inspect.",
                                   "N/A", component, {})

        try:
            train_df = pd.read_parquet(state.feature_artifact.engineered_train_path)
            target = state.profile.target_column if state.profile else None
            if not target or target not in train_df.columns:
                return self._make_item(1, check, "PASS",
                                       "Target column not found in engineered data.", "N/A",
                                       component, {})

            y = train_df[target]
            numeric_features = [
                c for c in train_df.columns
                if c != target and pd.api.types.is_numeric_dtype(train_df[c])
            ]

            high_corr: Dict[str, float] = {}
            for col in numeric_features:
                try:
                    corr = abs(float(train_df[col].corr(y)))
                    if corr >= _LEAKAGE_CORR_WARNING:
                        high_corr[col] = round(corr, 4)
                except Exception:
                    pass

            if not high_corr:
                return self._make_item(1, check, "PASS",
                                       "No feature exceeds Pearson |corr| ≥ 0.80 with target.",
                                       "None required.", component,
                                       {"max_corr": 0.0})

            max_corr = max(high_corr.values())
            status = "CRITICAL_FAIL" if max_corr >= _LEAKAGE_CORR_CRITICAL else "WARNING"
            obs = (
                f"{len(high_corr)} feature(s) show suspicious correlation with target: "
                f"{dict(list(high_corr.items())[:5])}. Max |corr|={max_corr:.4f}."
            )
            rec = (
                "Drop or inspect features with |corr| > 0.85 with target. "
                "These may encode the label directly. "
                "Rerun FeatureEngineeringAgent with drop_high_corr_features directive."
            )
            return self._make_item(1, check, status, obs, rec, component,
                                   {"high_corr_features": high_corr, "max_corr": max_corr})

        except Exception as exc:
            return self._make_item(1, check, "WARNING",
                                   f"Leakage check failed to execute: {exc}",
                                   "Inspect engineered data manually.", component, {})

    def _v2_contamination(self, state: DataScienceState) -> CriticAuditItem:
        """V2 – Distribution-shift check between train / val / test engineered features."""
        component = "FeatureEngineeringAgent"
        check = "Train/Val/Test Contamination"

        if not (state.feature_artifact and state.feature_artifact.engineered_train_path
                and state.feature_artifact.engineered_val_path):
            return self._make_item(2, check, "PASS",
                                   "Engineered splits not available for contamination check.",
                                   "N/A", component, {})

        try:
            train_df = pd.read_parquet(state.feature_artifact.engineered_train_path)
            val_df   = pd.read_parquet(state.feature_artifact.engineered_val_path)
            target   = state.profile.target_column if state.profile else None

            scaled_cols = [
                c for c in train_df.columns
                if c != target and c.endswith("_scaled")
            ]
            if not scaled_cols:
                return self._make_item(2, check, "PASS",
                                       "No scaled features found to inspect.", "N/A",
                                       component, {})

            # Scaled train features should have mean ≈ 0 (by construction of StandardScaler)
            bad_train_means: Dict[str, float] = {}
            for col in scaled_cols[:10]:                # check up to 10 columns
                m = abs(float(train_df[col].mean()))
                if m > _TRAIN_SCALED_MEAN_TOL:
                    bad_train_means[col] = round(m, 4)

            if bad_train_means:
                obs = (
                    f"Scaled train features have non-zero means (expected ≈ 0): "
                    f"{bad_train_means}. This suggests the scaler was NOT fitted "
                    f"exclusively on the training split."
                )
                rec = "Ensure StandardScaler.fit() is called only on X_train, never on X_val/X_test."
                return self._make_item(2, check, "WARNING", obs, rec, component,
                                       {"bad_train_means": bad_train_means})

            # Val means should be reasonably close to 0 (within ±1 std of train distribution)
            large_val_drift: Dict[str, float] = {}
            for col in scaled_cols[:10]:
                train_std = float(train_df[col].std()) or 1.0
                val_mean  = abs(float(val_df[col].mean()))
                if val_mean > max(3.0 * train_std, 0.5):
                    large_val_drift[col] = round(val_mean, 4)

            if large_val_drift:
                obs = (
                    f"Validation scaled features show large distributional drift from train: "
                    f"{large_val_drift}. Possible contamination or domain shift."
                )
                rec = "Check that val/test samples are drawn from the same distribution as train."
                return self._make_item(2, check, "WARNING", obs, rec, component,
                                       {"val_drift": large_val_drift})

            return self._make_item(2, check, "PASS",
                                   "Train/val scaled-feature distributions are consistent.",
                                   "None required.", component,
                                   {"checked_features": len(scaled_cols)})

        except Exception as exc:
            return self._make_item(2, check, "WARNING",
                                   f"Contamination check failed: {exc}",
                                   "Inspect engineered splits manually.", component, {})

    def _v3_overfitting(
        self, state: DataScienceState, candidates: List[TrainedModelCandidate]
    ) -> CriticAuditItem:
        """V3 – Overfitting (train–val gap) and generalisation gap (val–test gap)."""
        component = "MLModelAgent / EvaluationAgent"
        check = "Overfitting & Generalization Gap"

        report = state.evaluation_report
        if not report:
            return self._make_item(3, check, "PASS",
                                   "No evaluation report available yet.", "N/A", component, {})

        # Find champion's metrics
        champion = next((c for c in report.leaderboard if c.is_champion), None)
        if not champion:
            return self._make_item(3, check, "PASS",
                                   "No champion found in leaderboard.", "N/A", component, {})

        primary = "f1_macro" if "classification" in (state.profile.problem_type or "") else "r2"
        m = champion.metrics.get(primary)
        if not m:
            return self._make_item(3, check, "PASS",
                                   "Primary metric not in champion metrics.", "N/A", component, {})

        issues: List[str] = []
        status = "PASS"
        evidence: Dict[str, float] = {}

        train_score = m.train_score or 0.0
        val_score   = m.val_score   or 0.0
        test_score  = m.test_score  # may be None

        overfit_gap = train_score - val_score
        evidence["train_score"] = round(train_score, 4)
        evidence["val_score"]   = round(val_score,   4)
        evidence["overfit_gap"] = round(overfit_gap, 4)

        if overfit_gap >= _OVERFIT_GAP_CRITICAL:
            issues.append(f"Severe overfitting: train–val gap = {overfit_gap:.4f}")
            status = "CRITICAL_FAIL"
        elif overfit_gap >= _OVERFIT_GAP_WARNING:
            issues.append(f"Mild overfitting: train–val gap = {overfit_gap:.4f}")
            status = "WARNING"

        if test_score is not None:
            gen_gap = abs(val_score - test_score)
            evidence["test_score"]       = round(test_score, 4)
            evidence["generalization_gap"] = round(gen_gap,  4)
            if gen_gap >= _GEN_GAP_CRITICAL:
                issues.append(f"Large generalisation gap: |val–test| = {gen_gap:.4f}")
                if status != "CRITICAL_FAIL":
                    status = "CRITICAL_FAIL"
            elif gen_gap >= _GEN_GAP_WARNING:
                issues.append(f"Mild generalisation gap: |val–test| = {gen_gap:.4f}")
                if status == "PASS":
                    status = "WARNING"

        if not issues:
            return self._make_item(3, check, "PASS",
                                   f"No overfitting detected. train={train_score:.4f}, "
                                   f"val={val_score:.4f}.",
                                   "None required.", component, evidence)

        obs = "; ".join(issues)
        rec = (
            "Apply stronger regularisation: reduce model complexity, increase L2 penalty, "
            "add min_samples_leaf, reduce max_depth, or use early stopping. "
            "Rerun MLModelAgent with stronger_regularization directive."
        )
        return self._make_item(3, check, status, obs, rec, component, evidence)

    def _v4_class_imbalance(self, state: DataScienceState) -> CriticAuditItem:
        """V4 – Extreme class imbalance from dataset profile."""
        component = "DataAnalystAgent / MLModelAgent"
        check = "Extreme Class Imbalance"

        if not state.profile:
            return self._make_item(4, check, "PASS", "Profile not available.", "N/A",
                                   component, {})

        ratio = state.profile.class_imbalance_ratio
        ptype = state.profile.problem_type or ""

        if "regression" in ptype or ratio is None:
            return self._make_item(4, check, "PASS",
                                   "Regression task – imbalance check not applicable.",
                                   "N/A", component, {"ratio": None})

        evidence = {"class_imbalance_ratio": round(ratio, 2)}

        if ratio >= _IMBALANCE_CRITICAL:
            obs = (
                f"Extreme class imbalance detected: majority/minority ratio = {ratio:.1f}x. "
                f"Models trained on this data will likely be biased toward the majority class."
            )
            rec = (
                "Rerun MLModelAgent with enhanced_class_balance directive. "
                "Consider SMOTE oversampling, class_weight='balanced', or threshold tuning."
            )
            return self._make_item(4, check, "CRITICAL_FAIL", obs, rec, component, evidence)

        if ratio >= _IMBALANCE_WARNING:
            obs = f"Moderate class imbalance detected: ratio = {ratio:.1f}x."
            rec = "Ensure class_weight='balanced' is used. Monitor minority-class recall."
            return self._make_item(4, check, "WARNING", obs, rec, component, evidence)

        return self._make_item(4, check, "PASS",
                               f"Class imbalance ratio = {ratio:.1f}x – within acceptable range.",
                               "None required.", component, evidence)

    def _v5_high_val_performance(
        self, state: DataScienceState, candidates: List[TrainedModelCandidate]
    ) -> CriticAuditItem:
        """V5 – Suspiciously high validation score (potential target leakage)."""
        component = "FeatureEngineeringAgent / MLModelAgent"
        check = "Suspiciously High Validation Performance"

        if not candidates:
            return self._make_item(5, check, "PASS", "No candidates to inspect.", "N/A",
                                   component, {})

        primary = "f1_macro" if "classification" in (state.profile.problem_type or "") else "r2"
        val_scores = [
            c.metrics[primary].val_score
            for c in candidates
            if primary in c.metrics and c.metrics[primary].val_score is not None
        ]
        if not val_scores:
            return self._make_item(5, check, "PASS", "No val scores available.", "N/A",
                                   component, {})

        max_val = max(val_scores)
        evidence = {"max_val_score": round(max_val, 4), "metric": primary}

        if max_val >= _HIGH_VAL_CRITICAL:
            obs = (
                f"Validation {primary} = {max_val:.4f} is suspiciously high (≥ {_HIGH_VAL_CRITICAL}). "
                f"This strongly suggests target leakage – a feature encodes the label directly."
            )
            rec = (
                "Inspect features with |Pearson corr| > 0.85 with target. "
                "Rerun FeatureEngineeringAgent with drop_high_corr_features directive."
            )
            return self._make_item(5, check, "CRITICAL_FAIL", obs, rec, component, evidence)

        if max_val >= _HIGH_VAL_WARNING:
            obs = (
                f"Validation {primary} = {max_val:.4f} is unusually high (≥ {_HIGH_VAL_WARNING}). "
                f"Verify no near-duplicate of the target column exists in the features."
            )
            rec = "Double-check feature construction for indirect label encoding."
            return self._make_item(5, check, "WARNING", obs, rec, component, evidence)

        return self._make_item(5, check, "PASS",
                               f"Validation {primary} = {max_val:.4f} – within expected range.",
                               "None required.", component, evidence)

    def _v6_cv_stability(
        self, state: DataScienceState, candidates: List[TrainedModelCandidate]
    ) -> CriticAuditItem:
        """V6 – Cross-validation instability (high std across folds)."""
        component = "MLModelAgent"
        check = "Unstable Cross-Validation"

        if not candidates:
            return self._make_item(6, check, "PASS", "No candidates to inspect.", "N/A",
                                   component, {})

        primary = "f1_macro" if "classification" in (state.profile.problem_type or "") else "r2"
        cv_stds = {
            c.model_name: c.metrics[primary].cv_std
            for c in candidates
            if primary in c.metrics and c.metrics[primary].cv_std is not None
        }
        if not cv_stds:
            return self._make_item(6, check, "PASS", "CV std not available.", "N/A",
                                   component, {})

        max_std  = max(cv_stds.values())
        max_name = max(cv_stds, key=lambda k: cv_stds[k])
        evidence = {"max_cv_std": round(max_std, 4), "worst_model": max_name}

        if max_std >= _CV_STD_CRITICAL:
            obs = (
                f"Cross-validation std of '{max_name}' = {max_std:.4f} ≥ {_CV_STD_CRITICAL}. "
                f"High CV variance indicates the model's performance is strongly fold-dependent."
            )
            rec = (
                "Rerun MLModelAgent with increase_cv_folds directive. "
                "Use StratifiedKFold with n_splits ≥ 7 and consider regularised models."
            )
            return self._make_item(6, check, "CRITICAL_FAIL", obs, rec, component, evidence)

        if max_std >= _CV_STD_WARNING:
            obs = f"CV std of '{max_name}' = {max_std:.4f} ≥ {_CV_STD_WARNING} – moderate instability."
            rec = "Consider adding more CV folds or using more regularised models."
            return self._make_item(6, check, "WARNING", obs, rec, component, evidence)

        return self._make_item(6, check, "PASS",
                               f"All CV stds within tolerance (max = {max_std:.4f}).",
                               "None required.", component, evidence)

    def _v7_calibration(self, state: DataScienceState) -> CriticAuditItem:
        """V7 – Probability calibration quality via Brier score."""
        component = "EvaluationAgent"
        check = "Model Calibration"

        report = state.evaluation_report
        ptype  = state.profile.problem_type if state.profile else ""

        if "regression" in ptype:
            return self._make_item(7, check, "PASS",
                                   "Calibration check not applicable to regression.", "N/A",
                                   component, {})

        if not report or report.calibration_brier_score is None:
            return self._make_item(7, check, "PASS",
                                   "Brier score not available (model may lack predict_proba).",
                                   "N/A", component, {})

        brier = report.calibration_brier_score
        evidence = {"brier_score": round(brier, 4)}

        if brier >= _BRIER_CRITICAL:
            obs = (
                f"Brier score = {brier:.4f} ≥ {_BRIER_CRITICAL}. "
                f"The model's probability estimates are poorly calibrated – "
                f"confidences diverge significantly from empirical frequencies."
            )
            rec = (
                "Apply Platt scaling or isotonic regression via CalibratedClassifierCV. "
                "Rerun EvaluationAgent with apply_calibration directive."
            )
            return self._make_item(7, check, "CRITICAL_FAIL", obs, rec, component, evidence)

        if brier >= _BRIER_WARNING:
            obs = f"Brier score = {brier:.4f} ≥ {_BRIER_WARNING} – moderate miscalibration."
            rec = "Consider probability calibration if downstream decisions rely on confidence scores."
            return self._make_item(7, check, "WARNING", obs, rec, component, evidence)

        return self._make_item(7, check, "PASS",
                               f"Brier score = {brier:.4f} – calibration is acceptable.",
                               "None required.", component, evidence)

    def _v8_preprocessing(self, state: DataScienceState) -> CriticAuditItem:
        """V8 – Preprocessing consistency and leakage guard."""
        component = "DataCleaningAgent / FeatureEngineeringAgent"
        check = "Preprocessing Consistency"

        issues: List[str] = []

        # Check cleaning artifact leakage guard flag
        if state.cleaning_artifact and not state.cleaning_artifact.leakage_guard_passed:
            issues.append("DataCleaningAgent: leakage_guard_passed=False – "
                          "imputer / scaler may have been fitted on val/test data.")

        # Check that the engineered paths actually exist
        fa = state.feature_artifact
        if fa:
            for label, path in [
                ("train", fa.engineered_train_path),
                ("val",   fa.engineered_val_path),
                ("test",  fa.engineered_test_path),
            ]:
                if path:
                    from pathlib import Path
                    if not Path(path).exists():
                        issues.append(f"Engineered {label} parquet file missing: {path}")

        if issues:
            obs = " | ".join(issues)
            rec = (
                "Ensure all fitting (imputers, scalers, encoders) is performed exclusively on "
                "the training split and applied (transform-only) to val/test."
            )
            # Leakage guard failure is a critical issue
            status = "CRITICAL_FAIL" if "leakage_guard_passed=False" in obs else "WARNING"
            return self._make_item(8, check, status, obs, rec, component,
                                   {"issues": issues})

        return self._make_item(8, check, "PASS",
                               "Leakage guard passed and all engineered splits are present.",
                               "None required.", component, {})

    # ──────────────────────────────────────────────────────────────────────────
    # Corrective planning
    # ──────────────────────────────────────────────────────────────────────────

    def _corrective_plan(
        self,
        audit_results: List[CriticAuditItem],
        state: DataScienceState,
    ) -> Tuple[Optional[str], str]:
        """
        Determines the target agent to rerun and produces a human-readable
        directive string from the CRITICAL_FAIL / WARNING items.

        Priority order: V1/V5 (leakage) → V8 (preprocessing) → V3 (overfit) →
                        V4 (imbalance) → V6 (CV) → V7 (calibration) → V2 (contamination)
        """
        failed = {a.vector_id: a for a in audit_results
                  if a.status in ("CRITICAL_FAIL", "WARNING")}

        if not failed:
            return None, "Pipeline passed all quality checks – no corrective action required."

        directive_parts: List[str] = []
        target_agent: Optional[str] = None

        # V1 / V5 → Feature leakage → drop high-corr features
        if 1 in failed or 5 in failed:
            target_agent = "FeatureEngineer"
            directive_parts.append(
                "DROP high-correlation features: rerun FeatureEngineeringAgent with "
                "drop_high_corr_features=True, corr_threshold=0.85."
            )

        # V8 → Preprocessing / contamination
        if 8 in failed and target_agent is None:
            target_agent = "FeatureEngineer"
            directive_parts.append(
                "FIX preprocessing leak: rerun FeatureEngineeringAgent ensuring strict "
                "train-only fitting."
            )

        # V3 → Overfitting
        if 3 in failed and target_agent is None:
            target_agent = "ModelTrainer"
            gap = failed[3].evidence.get("overfit_gap", "?")
            directive_parts.append(
                f"REDUCE overfitting (gap={gap}): rerun MLModelAgent with "
                f"stronger_regularization=True (lower max_depth, higher min_samples_leaf, L2)."
            )

        # V4 → Imbalance
        if 4 in failed and target_agent is None:
            target_agent = "ModelTrainer"
            ratio = failed[4].evidence.get("class_imbalance_ratio", "?")
            directive_parts.append(
                f"HANDLE imbalance (ratio={ratio}x): rerun MLModelAgent with "
                f"enhanced_class_balance=True."
            )

        # V6 → CV instability
        if 6 in failed and target_agent is None:
            target_agent = "ModelTrainer"
            std = failed[6].evidence.get("max_cv_std", "?")
            directive_parts.append(
                f"STABILISE cross-validation (cv_std={std}): rerun MLModelAgent with "
                f"increase_cv_folds=True, n_cv_splits=7."
            )

        # V7 → Calibration
        if 7 in failed and target_agent is None:
            target_agent = "Evaluator"
            directive_parts.append(
                "CALIBRATE probabilities: rerun EvaluationAgent with apply_calibration=True "
                "(Platt scaling via CalibratedClassifierCV)."
            )

        # V2 → Contamination (lower priority – already covered by FE if also V8)
        if 2 in failed and target_agent is None:
            target_agent = "FeatureEngineer"
            directive_parts.append(
                "FIX distribution shift: rerun FeatureEngineeringAgent with strict_train_only=True."
            )

        # Add warnings as supplementary notes
        warn_names = [a.check_name for a in audit_results if a.status == "WARNING"]
        if warn_names:
            directive_parts.append(f"[WARNING] Monitor: {', '.join(warn_names)}.")

        return target_agent, " | ".join(directive_parts)

    # ──────────────────────────────────────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_item(
        vector_id: int,
        check_name: str,
        status: str,
        observation: str,
        recommendation: str,
        affected_component: str,
        evidence: Dict[str, Any],
    ) -> CriticAuditItem:
        return CriticAuditItem(
            vector_id=vector_id,
            check_name=check_name,
            status=status,
            observation=observation,
            recommendation=recommendation,
            affected_component=affected_component,
            evidence=evidence,
        )
