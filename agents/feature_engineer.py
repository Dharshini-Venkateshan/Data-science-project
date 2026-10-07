"""
Feature Engineering Agent for Autonomous AI Data Scientist.
Transforms numeric distributions, encodes categoricals safely, groups high-cardinality features,
and strictly enforces train-only fitting to eliminate data leakage.
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from agents.base import BaseAgent
from core.state import DataScienceState, FeatureArtifact
from core.llm_client import BaseLLMClient
from core.config import WORKSPACE_DIR


class FeatureEngineeringAgent(BaseAgent):
    """
    Constructs feature representations strictly fitted on the training split
    and applied identically to validation and test splits.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        super().__init__(name="FeatureEngineeringAgent", llm_client=llm_client)

    def execute(self, state: DataScienceState) -> DataScienceState:
        """
        Executes feature transformation pipeline on train, validation, and test splits.
        """
        if not state.cleaning_artifact or not state.cleaning_artifact.clean_train_path:
            raise ValueError("FeatureEngineeringAgent requires cleaning_artifact from DataCleaningAgent.")

        # ── Read corrective directives set by Critic / Orchestrator ────────
        directives = state.critic_directives or {}

        # 1. Load Sanitized Data Splits
        train_df = pd.read_parquet(state.cleaning_artifact.clean_train_path)
        val_df = pd.read_parquet(state.cleaning_artifact.clean_val_path)
        test_df = pd.read_parquet(state.cleaning_artifact.clean_test_path)

        target_col = state.profile.target_column if state.profile else None
        if not target_col or target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in training dataset.")

        # Separate Features and Targets
        y_train = train_df[target_col].copy()
        y_val = val_df[target_col].copy()
        y_test = test_df[target_col].copy()

        X_train = train_df.drop(columns=[target_col]).copy()
        X_val = val_df.drop(columns=[target_col]).copy()
        X_test = test_df.drop(columns=[target_col]).copy()

        transformations_applied: List[str] = []
        dropped_cols: List[str] = []

        # ── Directive: drop features with high Pearson |corr| with target ──
        if directives.get("drop_high_corr_features"):
            corr_threshold = float(directives.get("corr_threshold", 0.85))
            cols_to_drop_corr: List[str] = []
            for col in X_train.columns:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    try:
                        if abs(float(X_train[col].corr(y_train))) >= corr_threshold:
                            cols_to_drop_corr.append(col)
                    except Exception:
                        pass
            if cols_to_drop_corr:
                X_train = X_train.drop(columns=cols_to_drop_corr)
                X_val   = X_val.drop(columns=cols_to_drop_corr, errors="ignore")
                X_test  = X_test.drop(columns=cols_to_drop_corr, errors="ignore")
                transformations_applied.append(
                    f"[Critic directive] Dropped {len(cols_to_drop_corr)} high-corr features "
                    f"(threshold={corr_threshold}): {cols_to_drop_corr}"
                )
                dropped_cols.extend(cols_to_drop_corr)
                state.log(
                    f"FeatureEngineeringAgent (corrective): dropped {cols_to_drop_corr} "
                    f"(|corr with target| >= {corr_threshold})."
                )

        # Identify Feature Types on Training Set
        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        cat_cols = [c for c in X_train.columns if c not in num_cols]

        X_train_processed_parts = []
        X_val_processed_parts = []
        X_test_processed_parts = []

        # 2. Numerical Feature Transformations
        if num_cols:
            # Check for high positive skewness on Train only
            skewed_num_cols = []
            for col in num_cols:
                if (X_train[col] >= 0).all():
                    skew_val = X_train[col].skew()
                    if skew_val > 1.5:
                        skewed_num_cols.append(col)
                        X_train[col] = np.log1p(X_train[col])
                        X_val[col] = np.log1p(X_val[col].clip(lower=0))
                        X_test[col] = np.log1p(X_test[col].clip(lower=0))

            if skewed_num_cols:
                transformations_applied.append(f"Applied log1p transform on skewed numeric features: {skewed_num_cols}")

            # Standard Scaling fitted strictly on X_train
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(X_train[num_cols])
            val_scaled = scaler.transform(X_val[num_cols])
            test_scaled = scaler.transform(X_test[num_cols])

            df_train_num = pd.DataFrame(train_scaled, columns=[f"{c}_scaled" for c in num_cols], index=X_train.index)
            df_val_num = pd.DataFrame(val_scaled, columns=[f"{c}_scaled" for c in num_cols], index=X_val.index)
            df_test_num = pd.DataFrame(test_scaled, columns=[f"{c}_scaled" for c in num_cols], index=X_test.index)

            X_train_processed_parts.append(df_train_num)
            X_val_processed_parts.append(df_val_num)
            X_test_processed_parts.append(df_test_num)
            transformations_applied.append(f"StandardScaled {len(num_cols)} numerical features.")

        # 3. Categorical Feature Encoding
        if cat_cols:
            # Handle high-cardinality by grouping infrequent categories on Train (< 2% occurrence)
            for c in cat_cols:
                val_freqs = X_train[c].value_counts(normalize=True)
                rare_cats = set(val_freqs[val_freqs < 0.02].index)
                if rare_cats:
                    X_train[c] = X_train[c].apply(lambda x: "rare_other" if x in rare_cats else x)
                    X_val[c] = X_val[c].apply(lambda x: "rare_other" if x in rare_cats else x)
                    X_test[c] = X_test[c].apply(lambda x: "rare_other" if x in rare_cats else x)
                    transformations_applied.append(f"Grouped {len(rare_cats)} rare categories into 'rare_other' for '{c}'.")

            # OneHotEncoder fitted strictly on Train
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            train_encoded = ohe.fit_transform(X_train[cat_cols].astype(str))
            val_encoded = ohe.transform(X_val[cat_cols].astype(str))
            test_encoded = ohe.transform(X_test[cat_cols].astype(str))

            encoded_feature_names = ohe.get_feature_names_out(cat_cols)
            df_train_cat = pd.DataFrame(train_encoded, columns=encoded_feature_names, index=X_train.index)
            df_val_cat = pd.DataFrame(val_encoded, columns=encoded_feature_names, index=X_val.index)
            df_test_cat = pd.DataFrame(test_encoded, columns=encoded_feature_names, index=X_test.index)

            X_train_processed_parts.append(df_train_cat)
            X_val_processed_parts.append(df_val_cat)
            X_test_processed_parts.append(df_test_cat)
            transformations_applied.append(f"One-hot encoded {len(cat_cols)} categorical features into {len(encoded_feature_names)} dummy columns.")

        # 4. Concatenate Engineered Feature Matrices
        final_X_train = pd.concat(X_train_processed_parts, axis=1)
        final_X_val = pd.concat(X_val_processed_parts, axis=1)
        final_X_test = pd.concat(X_test_processed_parts, axis=1)

        # Drop zero-variance features if any
        zero_var_cols = [c for c in final_X_train.columns if final_X_train[c].var() == 0]
        if zero_var_cols:
            final_X_train = final_X_train.drop(columns=zero_var_cols)
            final_X_val = final_X_val.drop(columns=zero_var_cols)
            final_X_test = final_X_test.drop(columns=zero_var_cols)
            dropped_cols.extend(zero_var_cols)
            transformations_applied.append(f"Removed {len(zero_var_cols)} zero-variance features.")

        # Re-attach target column
        final_train_df = pd.concat([final_X_train, y_train], axis=1)
        final_val_df = pd.concat([final_X_val, y_val], axis=1)
        final_test_df = pd.concat([final_X_test, y_test], axis=1)

        # 5. Persist Engineered Splits
        processed_dir = (WORKSPACE_DIR / state.session_id / "processed").resolve()
        processed_dir.mkdir(parents=True, exist_ok=True)

        eng_train_path = processed_dir / "train_engineered.parquet"
        eng_val_path = processed_dir / "val_engineered.parquet"
        eng_test_path = processed_dir / "test_engineered.parquet"

        final_train_df.to_parquet(eng_train_path, index=False)
        final_val_df.to_parquet(eng_val_path, index=False)
        final_test_df.to_parquet(eng_test_path, index=False)

        engineered_feature_names = [c for c in final_train_df.columns if c != target_col]

        # 6. Record Structured Feature Artifact
        code_summary = (
            "# Reproducible Feature Engineering Pipeline\n"
            "# 1. Skewness log1p transform on positive continuous features\n"
            "# 2. StandardScaler fit on Train, transformed Val & Test\n"
            "# 3. Rare category aggregation & OneHotEncoder fit on Train\n"
            f"# Engineered Feature Count: {len(engineered_feature_names)}\n"
        )

        artifact = FeatureArtifact(
            code=code_summary,
            engineered_feature_names=engineered_feature_names,
            dropped_feature_names=dropped_cols,
            selection_method="Log1p Skew Transform + StandardScaler + OHE",
            total_features_count=len(engineered_feature_names),
            engineered_train_path=str(eng_train_path),
            engineered_val_path=str(eng_val_path),
            engineered_test_path=str(eng_test_path),
            notes="All scalers and encoders fitted exclusively on Train split."
        )

        state.feature_artifact = artifact
        state.status = "FEATURE_ENGINEERING"
        state.log(
            f"FeatureEngineeringAgent: Engineered {len(engineered_feature_names)} features. "
            f"Train shape={final_train_df.shape}, Val shape={final_val_df.shape}, Test shape={final_test_df.shape}."
        )

        return state
