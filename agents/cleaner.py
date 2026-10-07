"""
Data Cleaning Agent for Autonomous AI Data Scientist.
Implements reproducible data sanitization, deduplication, train/val/test splitting,
and train-only fitted imputation to strictly prevent data leakage.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from agents.base import BaseAgent
from core.state import DataScienceState, CleaningArtifact
from core.llm_client import BaseLLMClient
from core.config import WORKSPACE_DIR


class DataCleaningAgent(BaseAgent):
    """
    Sanitizes raw data and partitions into Train, Validation, and Test splits.
    Strictly fits imputers on the training set to prevent leakage.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        super().__init__(name="DataCleaningAgent", llm_client=llm_client)

    def execute(self, state: DataScienceState) -> DataScienceState:
        """
        Executes end-to-end cleaning pipeline and persists sanitized splits.
        """
        if not state.profile:
            raise ValueError("DataCleaningAgent requires an existing DataProfile in state.")

        raw_path = Path(state.raw_dataset_path)
        df = pd.read_csv(raw_path, low_memory=False)
        profile = state.profile
        target_col = profile.target_column

        transformations_applied: List[str] = []
        dropped_columns: List[str] = []

        # 1. Deduplication
        initial_rows = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dedup_count = initial_rows - len(df)
        if dedup_count > 0:
            transformations_applied.append(f"Removed {dedup_count} exact duplicate rows.")

        # 2. Identify Non-Predictive Columns to Drop
        cols_to_drop = set()

        # Constant or near-constant columns
        for col in profile.constant_columns:
            if col != target_col and col in df.columns:
                cols_to_drop.add(col)
                transformations_applied.append(f"Dropped constant/near-constant column '{col}'.")

        # Pure identifiers
        for col_name, meta in profile.columns.items():
            if col_name != target_col and col_name in df.columns:
                if meta.inferred_type == "identifier":
                    cols_to_drop.add(col_name)
                    transformations_applied.append(f"Dropped identifier column '{col_name}'.")
                elif meta.missing_percentage >= 95.0:
                    cols_to_drop.add(col_name)
                    transformations_applied.append(f"Dropped column '{col_name}' (>95% missing values).")

        dropped_columns = sorted(list(cols_to_drop))
        if dropped_columns:
            df = df.drop(columns=dropped_columns)

        # 3. Handle Rows with Missing Target Values
        if target_col and target_col in df.columns:
            target_nulls = df[target_col].isnull().sum()
            if target_nulls > 0:
                df = df.dropna(subset=[target_col]).reset_index(drop=True)
                transformations_applied.append(f"Dropped {target_nulls} rows with null target values.")

        # 4. Strict Train / Validation / Test Splitting (70% / 15% / 15%)
        # Splitting happens BEFORE any imputation or feature transformations
        stratify_col = None
        if target_col and target_col in df.columns and "classification" in profile.problem_type:
            class_counts = df[target_col].value_counts()
            if (class_counts >= 2).all():
                stratify_col = df[target_col]

        # First split: 70% Train, 30% Temp (Val + Test)
        try:
            train_df, temp_df = train_test_split(
                df, test_size=0.30, random_state=42, stratify=stratify_col
            )
        except ValueError:
            # Fall back to unstratified if minority classes are too small
            train_df, temp_df = train_test_split(
                df, test_size=0.30, random_state=42, stratify=None
            )

        # Second split: Split Temp 50/50 -> 15% Val, 15% Test
        temp_stratify = None
        if target_col and target_col in temp_df.columns and "classification" in profile.problem_type:
            temp_counts = temp_df[target_col].value_counts()
            if (temp_counts >= 2).all():
                temp_stratify = temp_df[target_col]

        try:
            val_df, test_df = train_test_split(
                temp_df, test_size=0.50, random_state=42, stratify=temp_stratify
            )
        except ValueError:
            val_df, test_df = train_test_split(
                temp_df, test_size=0.50, random_state=42, stratify=None
            )

        transformations_applied.append(
            f"Partitioned data into Train (70%: {len(train_df)} rows), "
            f"Validation (15%: {len(val_df)} rows), and Test (15%: {len(test_df)} rows)."
        )

        # 5. Missing Value Imputation (Fitted STRICTLY on Train)
        feature_cols = [c for c in train_df.columns if c != target_col]
        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
        cat_cols = [c for c in feature_cols if c not in num_cols]

        # Numeric Imputation (Median)
        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
            val_df[num_cols] = num_imputer.transform(val_df[num_cols])
            test_df[num_cols] = num_imputer.transform(test_df[num_cols])
            transformations_applied.append(f"Applied median imputation on {len(num_cols)} numeric features.")

        # Categorical Imputation (Most Frequent)
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols].astype(str))
            val_df[cat_cols] = cat_imputer.transform(val_df[cat_cols].astype(str))
            test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols].astype(str))
            transformations_applied.append(f"Applied mode imputation on {len(cat_cols)} categorical features.")

        # 6. Persist Processed Splits
        processed_dir = (WORKSPACE_DIR / state.session_id / "processed").resolve()
        processed_dir.mkdir(parents=True, exist_ok=True)

        train_path = processed_dir / "train.parquet"
        val_path = processed_dir / "val.parquet"
        test_path = processed_dir / "test.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        # 7. Synthesize Pipeline Code Representation
        code_summary = (
            "# Deterministic Data Cleaning Pipeline\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.impute import SimpleImputer\n"
            "# 1. Deduplication\n"
            "df = df.drop_duplicates().reset_index(drop=True)\n"
            f"# 2. Dropped non-predictive columns: {dropped_columns}\n"
            "# 3. Stratified 70/15/15 train/val/test split\n"
            "# 4. Fit median/mode imputers on train only\n"
        )

        artifact = CleaningArtifact(
            code=code_summary,
            transformations_applied=transformations_applied,
            dropped_columns=dropped_columns,
            train_shape=list(train_df.shape),
            val_shape=list(val_df.shape),
            test_shape=list(test_df.shape),
            clean_train_path=str(train_path),
            clean_val_path=str(val_path),
            clean_test_path=str(test_path),
            leakage_guard_passed=True,
            notes="Imputers fitted strictly on training split; no test contamination."
        )

        state.cleaning_artifact = artifact
        state.status = "CLEANING"
        state.log(
            f"DataCleaningAgent: Sanitized dataset. Train shape={train_df.shape}, "
            f"Val shape={val_df.shape}, Test shape={test_df.shape}. "
            f"Dropped {len(dropped_columns)} columns."
        )

        return state
