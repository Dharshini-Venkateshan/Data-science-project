"""
Data Analyst Agent for Autonomous AI Data Scientist.
Profiles dataset schema, infers semantics, flags anomalies, and identifies target/problem type.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from pydantic import BaseModel

from agents.base import BaseAgent
from core.state import DataScienceState, DataProfile
from core.llm_client import BaseLLMClient
from tools.profiler import DataProfiler


class ProblemInferenceSchema(BaseModel):
    """Schema for LLM confirmation of target and problem type."""
    target_column: str
    problem_type: str
    reasoning: str


class DataAnalystAgent(BaseAgent):
    """
    Analyzes raw dataset properties, infers data types, detects class imbalance,
    and screens for potential leakage columns.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        super().__init__(name="DataAnalystAgent", llm_client=llm_client)

    def execute(self, state: DataScienceState) -> DataScienceState:
        """
        Profiles the dataset at state.raw_dataset_path and stores profile in state.
        """
        raw_path = Path(state.raw_dataset_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

        # Load raw data
        df = pd.read_csv(raw_path, low_memory=False)

        # 1. Deterministic Profiling
        profile = DataProfiler.profile_dataframe(
            df=df,
            target_column=state.plan.target_column if state.plan else None,
            user_goal=state.user_goal
        )

        # 2. LLM Reasoning Enhancement (if LLM is available and target is ambiguous)
        if self.llm_client and profile.target_column is None:
            try:
                col_summary = [f"{col}: dtype={meta.dtype}, unique={meta.unique_count}"
                               for col, meta in list(profile.columns.items())[:20]]
                prompt = (
                    f"User Objective: '{state.user_goal}'\n"
                    f"Columns: {col_summary}\n"
                    f"Identify the target column and the problem type (binary_classification, "
                    f"multiclass_classification, regression, or time_series)."
                )
                inference: ProblemInferenceSchema = self.llm_client.generate_structured(
                    prompt=prompt,
                    response_schema=ProblemInferenceSchema
                )
                if inference.target_column in df.columns:
                    profile.target_column = inference.target_column
                    profile.problem_type = inference.problem_type
            except Exception:
                pass  # Fall back gracefully to deterministic heuristics

        # Update State
        state.profile = profile
        state.status = "PROFILING"
        state.log(
            f"DataAnalystAgent: Profiled {profile.row_count} rows, {profile.col_count} columns. "
            f"Target='{profile.target_column}', ProblemType='{profile.problem_type}', "
            f"Duplicates={profile.duplicate_rows_count}, Constants={len(profile.constant_columns)}."
        )

        return state
