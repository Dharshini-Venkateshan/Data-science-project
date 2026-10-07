"""
Statistical Data Profiler for Autonomous AI Data Scientist.
Performs deterministic column type inference, anomaly detection, missingness analysis,
duplicate auditing, class imbalance detection, and target leakage screening.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from core.state import ColumnMetadata, DataProfile


class DataProfiler:
    """
    Analyzes raw or ingested DataFrames and generates strongly-typed DataProfile artifacts.
    """

    ID_NAME_PATTERNS = re.compile(r"(^id$|_id$|^uuid$|^guid$|^index$|^row_id$|^record_id$)", re.IGNORECASE)

    @classmethod
    def infer_column_type(cls, series: pd.Series, col_name: str) -> str:
        """
        Determines the semantic data type of a column using dtype inspection and heuristics.
        """
        total_count = len(series)
        valid_series = series.dropna()
        unique_count = valid_series.nunique()

        if total_count == 0 or unique_count == 0:
            return "unknown"

        # 1. Identifier check
        if cls.ID_NAME_PATTERNS.search(col_name) and (unique_count / total_count > 0.8):
            return "identifier"
        if unique_count == total_count and pd.api.types.is_string_dtype(series):
            return "identifier"

        # 2. Datetime check
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_string_dtype(series) and unique_count > 10:
            # Check a small sample to see if it parses as datetime
            sample = valid_series.head(20).astype(str)
            try:
                pd.to_datetime(sample, format="mixed")
                # If sample parsed without error, verify with a larger chunk
                return "datetime"
            except (ValueError, TypeError):
                pass

        # 3. Numeric checks
        if pd.api.types.is_numeric_dtype(series):
            # Check if boolean or binary numeric
            if unique_count <= 2:
                return "categorical_nominal"
            # Discrete vs continuous
            if pd.api.types.is_integer_dtype(series) and unique_count < 25:
                return "numeric_discrete"
            return "numeric_continuous"

        # 4. Categorical checks
        if pd.api.types.is_bool_dtype(series):
            return "categorical_nominal"

        # Default string/object
        if unique_count < 20:
            return "categorical_nominal"
        elif unique_count / total_count > 0.6:
            return "text"
        else:
            return "categorical_nominal"

    @classmethod
    def detect_problem_type(cls, df: pd.DataFrame, target_col: Optional[str]) -> str:
        """
        Infers the machine learning problem type from the target column characteristics.
        """
        if not target_col or target_col not in df.columns:
            return "unknown"

        target_series = df[target_col].dropna()
        n_unique = target_series.nunique()

        if n_unique == 2:
            return "binary_classification"
        elif 3 <= n_unique <= 20:
            return "multiclass_classification"
        elif pd.api.types.is_numeric_dtype(target_series):
            return "regression"
        else:
            return "multiclass_classification"

    @classmethod
    def profile_dataframe(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        user_goal: Optional[str] = None
    ) -> DataProfile:
        """
        Conducts comprehensive profiling of a DataFrame and returns a DataProfile object.
        """
        row_count, col_count = df.shape
        columns_meta: Dict[str, ColumnMetadata] = {}
        high_cardinality_cols: List[str] = []
        constant_cols: List[str] = []
        skewed_cols: List[str] = []
        potential_leakage_cols: List[str] = []

        # 1. Infer Target Column if not explicitly specified
        if not target_column and user_goal:
            # Check if user goal explicitly names columns, picking the earliest mentioned
            matches = []
            for col in df.columns:
                pattern = r"\b" + re.escape(str(col)) + r"\b"
                m = re.search(pattern, user_goal, re.IGNORECASE)
                if m:
                    matches.append((m.start(), str(col)))
            if matches:
                matches.sort(key=lambda x: x[0])
                target_column = matches[0][1]
        
        # Fallback heuristic: check for common target names if still None
        if not target_column:
            common_targets = ["target", "label", "class", "churn", "survived", "price", "income", "outcome"]
            for ct in common_targets:
                for col in df.columns:
                    if str(col).lower() == ct:
                        target_column = str(col)
                        break
                if target_column:
                    break

        problem_type = cls.detect_problem_type(df, target_column)
        imbalance_ratio: Optional[float] = None

        # 2. Check Class Imbalance
        if target_column and target_column in df.columns and "classification" in problem_type:
            val_counts = df[target_column].value_counts(dropna=True)
            if len(val_counts) >= 2:
                max_count = val_counts.iloc[0]
                min_count = val_counts.iloc[-1]
                imbalance_ratio = float(max_count / min_count) if min_count > 0 else None

        # 3. Analyze Column-Level Metadata
        for col in df.columns:
            series = df[col]
            missing_count = int(series.isnull().sum())
            missing_pct = float((missing_count / row_count) * 100.0) if row_count > 0 else 0.0
            unique_count = int(series.nunique(dropna=True))
            inferred_type = cls.infer_column_type(series, str(col))

            # Sample values for state tracking (up to 5 representative values)
            sample_vals = series.dropna().head(5).tolist()

            # Constant / Near-Constant Check
            if unique_count <= 1:
                constant_cols.append(str(col))
            elif row_count > 100:
                top_freq = series.value_counts(dropna=False).iloc[0]
                if (top_freq / row_count) >= 0.99:
                    constant_cols.append(str(col))

            # High Cardinality Check (for categorical/text)
            if inferred_type in ["categorical_nominal", "text"] and unique_count > 50:
                high_cardinality_cols.append(str(col))

            # Skewness check for continuous numeric features
            if inferred_type == "numeric_continuous" and unique_count > 10:
                numeric_series = pd.to_numeric(series.dropna(), errors="coerce")
                if len(numeric_series) > 10:
                    skew_val = float(numeric_series.skew())
                    if abs(skew_val) > 1.5:
                        skewed_cols.append(str(col))

            # Leakage / Suspicious Association Check
            if target_column and str(col) != target_column:
                # Direct name variant leakage (e.g. target_binary, is_target)
                if target_column.lower() in str(col).lower():
                    potential_leakage_cols.append(str(col))
                # Correlation leakage
                elif pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(df[target_column]):
                    valid_mask = series.notnull() & df[target_column].notnull()
                    if valid_mask.sum() > 20:
                        corr = abs(series[valid_mask].corr(df[target_column][valid_mask]))
                        if corr >= 0.98:
                            potential_leakage_cols.append(str(col))
                # Pure ID column used as feature
                elif inferred_type == "identifier":
                    potential_leakage_cols.append(str(col))

            columns_meta[str(col)] = ColumnMetadata(
                name=str(col),
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_percentage=round(missing_pct, 2),
                unique_count=unique_count,
                sample_values=sample_vals
            )

        # 4. Check Duplicate Rows
        duplicate_rows_count = int(df.duplicated().sum())

        return DataProfile(
            row_count=row_count,
            col_count=col_count,
            columns=columns_meta,
            target_column=target_column,
            problem_type=problem_type,
            class_imbalance_ratio=round(imbalance_ratio, 2) if imbalance_ratio is not None else None,
            duplicate_rows_count=duplicate_rows_count,
            constant_columns=constant_cols,
            potential_leakage_columns=list(set(potential_leakage_cols)),
            high_cardinality_columns=high_cardinality_cols,
            skewed_numeric_columns=skewed_cols
        )
