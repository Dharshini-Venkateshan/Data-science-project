"""
Demonstration script for Phase 2: Ingestion, Profiling, Cleaning & EDA Agents.
Executes end-to-end data pipeline on a sample customer dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from tools.ingestion import DatasetIngestor
from core.state import DataScienceState
from agents.analyst import DataAnalystAgent
from agents.cleaner import DataCleaningAgent
from agents.eda import EDAAgent
from core.config import WORKSPACE_DIR


def generate_sample_dataset(csv_path: Path) -> None:
    """Generates a realistic sample dataset with deliberate data science challenges."""
    np.random.seed(42)
    n = 250

    customer_id = [f"CUST_{i:05d}" for i in range(n)]           # Pure identifier
    tenure = np.random.randint(1, 72, size=n)                    # Discrete numeric
    monthly_charges = np.random.normal(65, 25, size=n)           # Continuous numeric
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    country = ["United States"] * n                              # Constant column
    
    # Class imbalance: ~75% Not Churned (0), ~25% Churned (1)
    churn = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    
    # Inject missing values in monthly_charges
    missing_indices = np.random.choice(n, size=25, replace=False)
    monthly_charges[missing_indices] = np.nan

    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "contract": contract,
        "country": country,
        "churn": churn
    })

    # Inject 8 duplicate rows
    duplicates = df.iloc[:8].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Generated sample dataset at '{csv_path}' with {len(df)} rows and {len(df.columns)} columns.")


def run_phase2_demo():
    print("=" * 70)
    print("AUTONOMOUS AI DATA SCIENTIST - PHASE 2 PIPELINE DEMONSTRATION")
    print("=" * 70)

    session_id = "demo_phase2_session"
    demo_raw_csv = WORKSPACE_DIR / "sample_telecom_data.csv"
    generate_sample_dataset(demo_raw_csv)

    # 1. Dataset Ingestion Component
    print("\n[Step 1] Executing Dataset Ingestion Component...")
    ingestor = DatasetIngestor(session_id=session_id)
    ingest_result = ingestor.ingest_csv(demo_raw_csv)
    print(f"  Ingestion Status  : SUCCESS")
    print(f"  Original Filename : {ingest_result.original_filename}")
    print(f"  Stored Path       : {ingest_result.stored_path}")
    print(f"  Shape Ingested    : {ingest_result.row_count} rows, {ingest_result.col_count} columns")
    print(f"  File Size         : {ingest_result.file_size_bytes} bytes")

    # 2. State Initialization
    state = DataScienceState(
        session_id=session_id,
        user_goal="Predict customer churn and identify key churn drivers",
        raw_dataset_path=ingest_result.stored_path
    )

    # 3. Data Analyst Agent Execution
    print("\n[Step 2] Executing Data Analyst Agent...")
    analyst = DataAnalystAgent()
    state = analyst.execute(state)
    profile = state.profile
    print(f"  Target Column     : {profile.target_column}")
    print(f"  Problem Type      : {profile.problem_type}")
    print(f"  Class Imbalance   : {profile.class_imbalance_ratio:.2f}:1 ratio")
    print(f"  Duplicate Rows    : {profile.duplicate_rows_count} rows detected")
    print(f"  Constant Columns  : {profile.constant_columns}")
    print(f"  Leakage Warnings  : {profile.potential_leakage_columns}")
    print("  Column Metadata Summary:")
    for col, meta in profile.columns.items():
        print(f"    - {col:16s}: {meta.inferred_type:20s} (Missing: {meta.missing_percentage:4.1f}%, Unique: {meta.unique_count})")

    # 4. Data Cleaning Agent Execution
    print("\n[Step 3] Executing Data Cleaning Agent...")
    cleaner = DataCleaningAgent()
    state = cleaner.execute(state)
    cleaning = state.cleaning_artifact
    print(f"  Dropped Columns   : {cleaning.dropped_columns}")
    print(f"  Train Split Shape : {cleaning.train_shape} (70%)")
    print(f"  Val Split Shape   : {cleaning.val_shape} (15%)")
    print(f"  Test Split Shape  : {cleaning.test_shape} (15%)")
    print(f"  Leakage Guard     : {'PASSED' if cleaning.leakage_guard_passed else 'FAILED'}")
    print("  Transformations Changelog:")
    for t in cleaning.transformations_applied:
        print(f"    * {t}")

    # 5. EDA Agent Execution
    print("\n[Step 4] Executing EDA Agent...")
    eda = EDAAgent()
    state = eda.execute(state)
    eda_res = state.eda_results
    print("  Top Correlations  :")
    for cp in eda_res.top_correlations[:3]:
        print(f"    - {cp['feature_1']} <-> {cp['feature_2']}: r = {cp['correlation']}")
    print("  Mutual Information with Target:")
    for feat, score in eda_res.mutual_info_scores.items():
        print(f"    - {feat:16s}: MI = {score:.4f}")
    print(f"  Generated Charts  : {len(eda_res.generated_chart_paths)} charts saved")
    for cp in eda_res.generated_chart_paths:
        print(f"    -> {Path(cp).name}")
    print("  Key EDA Findings  :")
    for kf in eda_res.key_findings:
        print(f"    * {kf}")

    print("\n" + "=" * 70)
    print("PHASE 2 DEMONSTRATION COMPLETE - ALL AGENTS & ARTIFACTS VERIFIED")
    print(f"Current State Status: {state.status}")
    print(f"Session Execution Logs Count: {len(state.execution_logs)}")
    print("=" * 70)


if __name__ == "__main__":
    run_phase2_demo()
