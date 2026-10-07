"""
Exploratory Data Analysis (EDA) Agent for Autonomous AI Data Scientist.
Computes correlations, mutual information scores, detects distribution anomalies,
and generates publication-ready visualization artifacts on the training split.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from agents.base import BaseAgent
from core.state import DataScienceState, EDAResults
from core.llm_client import BaseLLMClient
from tools.visualizer import Visualizer


class EDAAgent(BaseAgent):
    """
    Executes exploratory analysis strictly on the training set to prevent test leakage.
    Generates statistical metrics and saves visual charts as session artifacts.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        super().__init__(name="EDAAgent", llm_client=llm_client)

    def execute(self, state: DataScienceState) -> DataScienceState:
        """
        Executes EDA routines on the sanitized training split and updates state.
        """
        if not state.cleaning_artifact or not state.cleaning_artifact.clean_train_path:
            raise ValueError("EDAAgent requires clean_train_path from DataCleaningAgent.")

        train_path = Path(state.cleaning_artifact.clean_train_path)
        if not train_path.exists():
            raise FileNotFoundError(f"Cleaned training dataset not found at: {train_path}")

        df_train = pd.read_parquet(train_path)
        target_col = state.profile.target_column if state.profile else None
        problem_type = state.profile.problem_type if state.profile else "unknown"

        visualizer = Visualizer(session_id=state.session_id)
        generated_charts: List[str] = []
        statistical_anomalies: List[str] = []
        key_findings: List[str] = []

        # 1. Generate Missing Values Chart (from raw data preview for visualization)
        raw_df = pd.read_csv(state.raw_dataset_path, nrows=5000, low_memory=False)
        missing_chart = visualizer.plot_missing_values(raw_df)
        if missing_chart:
            generated_charts.append(missing_chart)

        # 2. Target Distribution Visualization
        if target_col and target_col in df_train.columns:
            target_chart = visualizer.plot_target_distribution(df_train, target_col, problem_type)
            if target_chart:
                generated_charts.append(target_chart)

        # 3. Numeric Feature Correlations
        numeric_df = df_train.select_dtypes(include=[np.number])
        top_correlations: List[Dict[str, Any]] = []

        if numeric_df.shape[1] >= 2:
            corr_chart = visualizer.plot_correlation_heatmap(numeric_df)
            if corr_chart:
                generated_charts.append(corr_chart)

            # Extract top absolute correlations between distinct feature pairs
            corr_matrix = numeric_df.corr().abs()
            unstacked = corr_matrix.unstack()
            # Filter out self-correlations (diagonal)
            unstacked = unstacked[
                unstacked.index.get_level_values(0) != unstacked.index.get_level_values(1)
            ].sort_values(ascending=False).drop_duplicates()
            
            for (f1, f2), val in unstacked.head(5).items():
                if val > 0.05:
                    top_correlations.append({
                        "feature_1": str(f1),
                        "feature_2": str(f2),
                        "correlation": round(float(val), 3)
                    })
                    if val > 0.85:
                        statistical_anomalies.append(
                            f"High multicollinearity detected between '{f1}' and '{f2}' (r={val:.2f})."
                        )

        # 4. Mutual Information with Target
        mutual_info_dict: Dict[str, float] = {}
        if target_col and target_col in df_train.columns:
            X = df_train.drop(columns=[target_col])
            y = df_train[target_col]

            # Use numeric features and encoded categoricals for MI calculation
            X_encoded = X.copy()
            for col in X_encoded.select_dtypes(include=["object", "category"]).columns:
                X_encoded[col] = X_encoded[col].astype("category").cat.codes

            numeric_mask = [col for col in X_encoded.columns if pd.api.types.is_numeric_dtype(X_encoded[col])]
            if numeric_mask:
                X_numeric = X_encoded[numeric_mask].fillna(0)
                try:
                    if "classification" in problem_type:
                        mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
                    else:
                        mi_scores = mutual_info_regression(X_numeric, y, random_state=42)

                    for col_name, score in zip(numeric_mask, mi_scores):
                        mutual_info_dict[col_name] = round(float(score), 4)

                    # Sort top predictive features by MI
                    sorted_mi = sorted(mutual_info_dict.items(), key=lambda x: x[1], reverse=True)
                    top_features = [f[0] for f in sorted_mi[:3]]
                    key_findings.append(f"Top informative features by Mutual Information: {', '.join(top_features)}.")

                    # Generate relationship plots for top 2 features vs target
                    for top_feat in top_features[:2]:
                        rel_chart = visualizer.plot_feature_vs_target(df_train, top_feat, target_col, problem_type)
                        if rel_chart:
                            generated_charts.append(rel_chart)

                except Exception as mi_err:
                    statistical_anomalies.append(f"Mutual Information calculation skipped: {mi_err}")

        # 5. Compile Summary Findings
        key_findings.append(f"Training split contains {len(df_train)} rows across {len(df_train.columns)} features.")
        if state.profile and state.profile.class_imbalance_ratio:
            key_findings.append(f"Target class imbalance ratio: {state.profile.class_imbalance_ratio:.2f}:1.")

        # Update State
        eda_results = EDAResults(
            top_correlations=top_correlations,
            mutual_info_scores=mutual_info_dict,
            generated_chart_paths=generated_charts,
            statistical_anomalies=statistical_anomalies,
            key_findings=key_findings
        )

        state.eda_results = eda_results
        state.status = "EDA"
        state.log(
            f"EDAAgent: Generated {len(generated_charts)} charts, identified "
            f"{len(top_correlations)} correlation pairs and {len(mutual_info_dict)} MI scores."
        )

        return state
