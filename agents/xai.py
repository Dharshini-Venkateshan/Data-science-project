"""
Explainability & XAI Agent for Autonomous AI Data Scientist (Phase 5).

Provides post-hoc model interpretability for the champion ML model:
- Tree-based models: SHAP TreeExplainer, global SHAP importance, SHAP summary plots.
- Linear models: Normalized coefficient importance, Linear SHAP / additive attributions.
- Local case explanations for representative validation samples.
- High-resolution artifact generation (feature_importance.png, shap_summary.png).
- Strict data integrity: explainers fit only on Train/Val background data; test set is NEVER used.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated pipelines
import matplotlib.pyplot as plt

import shap

from agents.base import BaseAgent
from core.state import DataScienceState, InsightArtifact, TrainedModelCandidate
from core.llm_client import BaseLLMClient
from core.config import WORKSPACE_DIR, RESEARCH_FIGURES_DIR


class XAIAgent(BaseAgent):
    """
    Explainability (XAI) Agent that inspects and interprets the champion model.
    Produces global importance charts, SHAP summary plots, and localized case breakdowns.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        super().__init__(name="XAIAgent", llm_client=llm_client)

    def execute(self, state: DataScienceState) -> DataScienceState:
        """Standard BaseAgent execution hook."""
        return self.explain(state)

    def explain(
        self,
        state: DataScienceState,
        candidates: Optional[List[TrainedModelCandidate]] = None,
        save_to_research: bool = True,
    ) -> DataScienceState:
        """
        Executes XAI analysis on the champion model and updates state.insight_artifact.

        Parameters
        ----------
        state            : Current DataScienceState
        candidates       : Optional list of TrainedModelCandidate objects
        save_to_research : If True, also copies key figures to research/figures/
        """
        if not state.feature_artifact or not state.feature_artifact.engineered_train_path:
            raise ValueError("XAIAgent requires an engineered feature artifact in state.")

        # 1. Identify champion model name and load model artifact
        champion_name = state.champion_model_name
        if not champion_name and state.evaluation_report and state.evaluation_report.leaderboard:
            champ_cand = next((c for c in state.evaluation_report.leaderboard if c.is_champion), None)
            if champ_cand:
                champion_name = champ_cand.model_name
                state.champion_model_name = champion_name

        if not champion_name:
            raise ValueError("No champion model identified in state to explain.")

        # Resolve model instance — always load from disk (TrainedModelCandidate
        # is a Pydantic schema and does NOT carry the live model object).
        model = None
        artifact_path: Optional[str] = None
        if candidates:
            for cand in candidates:
                if cand.model_name == champion_name:
                    artifact_path = cand.artifact_path
                    break

        if artifact_path and Path(artifact_path).exists():
            model = joblib.load(artifact_path)
        else:
            model_path = WORKSPACE_DIR / state.session_id / "models" / f"{champion_name}.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
            else:
                raise FileNotFoundError(f"Champion model file not found at '{model_path}'.")

        # 2. Load Train and Validation data splits (NEVER TEST SET)
        train_df = pd.read_parquet(state.feature_artifact.engineered_train_path)
        val_df = pd.read_parquet(state.feature_artifact.engineered_val_path)

        target_col = state.profile.target_column if state.profile else None
        if not target_col or target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in engineered data.")

        X_train = train_df.drop(columns=[target_col])
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]

        feature_names = list(X_train.columns)

        # 3. Setup output directories for visual artifacts
        session_figs_dir = WORKSPACE_DIR / state.session_id / "figures"
        session_figs_dir.mkdir(parents=True, exist_ok=True)

        feat_imp_path = session_figs_dir / "feature_importance.png"
        shap_summary_path = session_figs_dir / "shap_summary.png"

        # 4. Compute Feature Importance & SHAP Values based on model family
        is_linear = any(term in champion_name.lower() for term in ["logistic", "ridge", "linear", "lasso"])
        global_importance: Dict[str, float] = {}
        explanation_method = "LinearCoefficients" if is_linear else "TreeSHAP"
        shap_values_raw = None
        eval_sample = X_val.sample(min(100, len(X_val)), random_state=42)
        bg_sample = X_train.sample(min(100, len(X_train)), random_state=42)

        if is_linear:
            # ── Linear model: coefficient-based importance ──
            coef = getattr(model, "coef_", None)
            if coef is not None:
                if coef.ndim > 1:
                    raw_imp = np.mean(np.abs(coef), axis=0)
                else:
                    raw_imp = np.abs(coef)
                total = float(np.sum(raw_imp)) if float(np.sum(raw_imp)) > 0 else 1.0
                normalized = raw_imp / total
                for col, val in zip(feature_names, normalized):
                    global_importance[col] = round(float(val), 6)

            # Try Linear SHAP
            try:
                explainer = shap.LinearExplainer(model, bg_sample)
                shap_obj = explainer(eval_sample)
                shap_values_raw = shap_obj.values
                self._plot_shap_summary(shap_obj, eval_sample, shap_summary_path)
            except Exception as e:
                state.log(f"XAIAgent: LinearExplainer fallback note: {e}")
                self._create_coef_importance_plot(global_importance, shap_summary_path, title="Linear Feature Attributions")

        else:
            # ── Tree-based model: TreeSHAP importance ──
            try:
                explainer = shap.TreeExplainer(model, bg_sample)
                shap_obj = explainer(eval_sample)
                vals = shap_obj.values
                # For classification, values may be (N, D, C) or (N, D)
                if vals.ndim == 3:
                    # Pick positive class if binary, or average magnitude across classes
                    mean_abs = np.mean(np.abs(vals[:, :, 1] if vals.shape[2] == 2 else vals[:, :, 0]), axis=0)
                else:
                    mean_abs = np.mean(np.abs(vals), axis=0)

                total = float(np.sum(mean_abs)) if float(np.sum(mean_abs)) > 0 else 1.0
                norm_imp = mean_abs / total
                for col, val in zip(feature_names, norm_imp):
                    global_importance[col] = round(float(val), 6)

                shap_values_raw = vals
                self._plot_shap_summary(shap_obj, eval_sample, shap_summary_path)

            except Exception as e:
                state.log(f"XAIAgent: TreeExplainer fallback to feature_importances_: {e}")
                explanation_method = "NativeFeatureImportance"
                raw_imp = getattr(model, "feature_importances_", None)
                if raw_imp is not None:
                    total = float(np.sum(raw_imp)) if float(np.sum(raw_imp)) > 0 else 1.0
                    for col, val in zip(feature_names, raw_imp / total):
                        global_importance[col] = round(float(val), 6)
                self._create_coef_importance_plot(global_importance, shap_summary_path, title="Tree Feature Importances")

        # 5. Sort global feature importance in descending order
        global_importance = dict(
            sorted(global_importance.items(), key=lambda item: item[1], reverse=True)
        )

        # 6. Generate Global Feature Importance Bar Chart
        self._plot_feature_importance(global_importance, champion_name, feat_imp_path)

        # 7. Generate Local Case Explanations (3 representative samples from validation set)
        local_cases = self._generate_local_cases(
            model=model,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names,
            is_linear=is_linear,
            shap_values=shap_values_raw,
            eval_sample=eval_sample,
        )

        # 8. Analytical Vulnerability & Domain Checks
        vulnerability_warnings = []
        business_recommendations = []

        if global_importance:
            top_feature, top_weight = next(iter(global_importance.items()))
            if top_weight > 0.50:
                vulnerability_warnings.append(
                    f"Feature '{top_feature}' accounts for {top_weight*100:.1f}% of total model importance. "
                    "Verify it is not an inadvertent proxy or leaky attribute."
                )

        top_3 = list(global_importance.keys())[:3]
        business_recommendations.append(
            f"Primary predictive drivers identified: {', '.join(top_3)}. "
            "Prioritize monitoring and data quality safeguards on these input variables."
        )

        # 9. Optional LLM Executive Summary
        summary_text = (
            f"Model {champion_name} explained via {explanation_method}. "
            f"Top predictive features: {', '.join(list(global_importance.keys())[:5])}."
        )
        if self.llm_client:
            try:
                prompt = (
                    f"Explain champion model {champion_name} for business stakeholders:\n"
                    f"Top features and weights: {dict(list(global_importance.items())[:5])}\n"
                    f"Vulnerability warnings: {vulnerability_warnings}\n"
                    "Provide a 2-sentence executive summary of model behavior."
                )
                summary_text = self.llm_client.generate_text(prompt=prompt)
            except Exception:
                pass

        # 10. Copy key plots to research/figures if requested
        if save_to_research:
            try:
                import shutil
                if feat_imp_path.exists():
                    shutil.copy(feat_imp_path, RESEARCH_FIGURES_DIR / "feature_importance.png")
                if shap_summary_path.exists():
                    shutil.copy(shap_summary_path, RESEARCH_FIGURES_DIR / "shap_summary.png")
            except Exception as e:
                state.log(f"XAIAgent: Note copying figures to research/figures: {e}")

        # 11. Store structured artifact in state
        artifact = InsightArtifact(
            global_feature_importance=global_importance,
            feature_importance_plot_path=str(feat_imp_path),
            shap_summary_plot_path=str(shap_summary_path) if shap_summary_path.exists() else None,
            explanation_method=explanation_method,
            champion_model_name=champion_name,
            local_case_explanations=local_cases,
            vulnerability_warnings=vulnerability_warnings,
            business_recommendations=business_recommendations,
            summary_text=summary_text,
        )

        state.insight_artifact = artifact
        state.status = "EXPLAINING"
        state.log(
            f"XAIAgent: Generated explanations for champion '{champion_name}' "
            f"using {explanation_method}. Top feature: {list(global_importance.keys())[0]}."
        )
        return state

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Visualization & Local Case Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        model_name: str,
        output_path: Path,
        top_n: int = 12,
    ) -> None:
        """Generates a clean horizontal bar chart for top N features."""
        top_items = list(importance_dict.items())[:top_n]
        if not top_items:
            return

        features, scores = zip(*reversed(top_items))

        plt.figure(figsize=(9, max(4, len(features) * 0.4)))
        bars = plt.barh(features, scores, color="#2b5c8f", edgecolor="#1a365d")
        plt.xlabel("Relative Importance Score", fontsize=11, fontweight="bold")
        plt.title(f"Global Feature Importance ({model_name})", fontsize=13, fontweight="bold", pad=12)
        plt.grid(axis="x", linestyle="--", alpha=0.5)

        # Annotate bar values
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_shap_summary(self, shap_obj: Any, X_sample: pd.DataFrame, output_path: Path) -> None:
        """Renders and saves a SHAP summary plot."""
        plt.figure(figsize=(9, 6))
        try:
            shap.summary_plot(shap_obj, X_sample, show=False, max_display=10)
            plt.title("SHAP Summary Plot", fontsize=13, fontweight="bold", pad=12)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close()

    def _create_coef_importance_plot(
        self,
        importance_dict: Dict[str, float],
        output_path: Path,
        title: str = "Feature Importance Overview",
    ) -> None:
        """Fallback plot when SHAP summary plot cannot be rendered."""
        top_items = list(importance_dict.items())[:10]
        if not top_items:
            return
        features, scores = zip(*reversed(top_items))

        plt.figure(figsize=(9, 5))
        plt.barh(features, scores, color="#319795")
        plt.xlabel("Importance", fontsize=11)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _generate_local_cases(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: List[str],
        is_linear: bool,
        shap_values: Optional[np.ndarray],
        eval_sample: pd.DataFrame,
        n_cases: int = 3,
    ) -> List[Dict[str, Any]]:
        """Extracts top positive and negative feature contributions for representative cases."""
        local_cases = []
        indices_to_explain = list(range(min(n_cases, len(X_val))))

        for idx in indices_to_explain:
            row_features = X_val.iloc[idx]
            actual = y_val.iloc[idx] if idx < len(y_val) else None

            # Get prediction
            pred_val = None
            pred_prob = None
            try:
                pred = model.predict(X_val.iloc[[idx]])[0]
                pred_val = float(pred) if hasattr(pred, "__float__") else int(pred)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_val.iloc[[idx]])[0]
                    pred_prob = round(float(probs[1] if len(probs) > 1 else probs[0]), 4)
            except Exception:
                pass

            # Determine contributions
            contributions: Dict[str, float] = {}
            if is_linear and hasattr(model, "coef_"):
                coef = model.coef_
                c_vec = coef[0] if coef.ndim > 1 else coef
                for f_name, f_val, c_val in zip(feature_names, row_features, c_vec):
                    contributions[f_name] = round(float(f_val * c_val), 4)
            elif shap_values is not None and idx < len(eval_sample):
                sample_vals = shap_values[idx]
                if sample_vals.ndim == 2:
                    sample_vals = sample_vals[:, 1] if sample_vals.shape[1] == 2 else sample_vals[:, 0]
                for f_name, s_val in zip(feature_names, sample_vals):
                    contributions[f_name] = round(float(s_val), 4)
            else:
                # Default heuristic: feature value times sign of correlation
                for f_name, f_val in zip(feature_names[:5], row_features[:5]):
                    contributions[f_name] = round(float(f_val), 4)

            # Sort top positive and top negative
            sorted_pos = sorted(
                [(k, v) for k, v in contributions.items() if v > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            sorted_neg = sorted(
                [(k, v) for k, v in contributions.items() if v < 0],
                key=lambda x: x[1],
            )[:3]

            local_cases.append({
                "sample_index": int(idx),
                "actual_label": int(actual) if isinstance(actual, (np.integer, int)) else (float(actual) if actual is not None else None),
                "predicted_label": pred_val,
                "predicted_probability": pred_prob,
                "top_positive_features": [{"feature": k, "contribution": v} for k, v in sorted_pos],
                "top_negative_features": [{"feature": k, "contribution": v} for k, v in sorted_neg],
            })

        return local_cases


# Alias for semantic compatibility with InsightAgent
InsightAgent = XAIAgent
