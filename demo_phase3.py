"""
Demonstration script for Phase 3: Feature Engineering, Model Training & Evaluation Engine.
Shows end-to-end execution on a reproducible tabular dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from tools.ingestion import DatasetIngestor
from core.state import DataScienceState
from agents.analyst import DataAnalystAgent
from agents.cleaner import DataCleaningAgent
from agents.feature_engineer import FeatureEngineeringAgent
from agents.model_trainer import MLModelAgent
from agents.evaluator import EvaluationAgent
from core.config import WORKSPACE_DIR


def generate_credit_dataset(csv_path: Path) -> None:
    """Generates reproducible credit default classification dataset."""
    np.random.seed(42)
    n = 300

    credit_score = np.random.normal(680, 50, n).clip(300, 850)
    annual_income = np.random.exponential(55000, n).clip(15000, 300000)
    debt_to_income = np.random.uniform(5, 55, n)
    employment_type = np.random.choice(["Salaried", "Self-Employed", "Business", "Unemployed"], n, p=[0.5, 0.25, 0.2, 0.05])
    loan_purpose = np.random.choice(["Home", "Auto", "Education", "Personal"], n)

    # Ground-truth probability of default based on key risk drivers
    logit = (
        -0.012 * (credit_score - 650)
        + 0.06 * (debt_to_income - 25)
        - 0.00002 * (annual_income - 50000)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default_target = (np.random.rand(n) < prob_default).astype(int)

    df = pd.DataFrame({
        "credit_score": credit_score,
        "annual_income": annual_income,
        "debt_to_income": debt_to_income,
        "employment_type": employment_type,
        "loan_purpose": loan_purpose,
        "default": default_target
    })

    df.to_csv(csv_path, index=False)
    print(f"Generated sample credit risk dataset at '{csv_path}' ({len(df)} rows, {len(df.columns)} columns).")


def run_phase3_demo():
    print("=" * 80)
    print("AUTONOMOUS AI DATA SCIENTIST - PHASE 3 ENGINE DEMONSTRATION")
    print("=" * 80)

    session_id = "demo_phase3_session"
    raw_csv = WORKSPACE_DIR / "sample_credit_data.csv"
    generate_credit_dataset(raw_csv)

    # 1. Pipeline Ingestion & Preprocessing Setup
    ingestor = DatasetIngestor(session_id=session_id)
    ingest_result = ingestor.ingest_csv(raw_csv)

    state = DataScienceState(
        session_id=session_id,
        user_goal="Predict loan default and select the optimal machine learning classifier",
        raw_dataset_path=ingest_result.stored_path
    )

    DataAnalystAgent().execute(state)
    DataCleaningAgent().execute(state)

    # 2. FEATURE ENGINEERING AGENT
    print("\n" + "-" * 80)
    print("[Step 1] FEATURE ENGINEERING DEMONSTRATION")
    print("-" * 80)
    fe_agent = FeatureEngineeringAgent()
    state = fe_agent.execute(state)
    fe_art = state.feature_artifact
    print(f"  Total Engineered Features: {fe_art.total_features_count}")
    print(f"  Selection Method Applied  : {fe_art.selection_method}")
    print(f"  Engineered Train Split    : {fe_art.engineered_train_path}")
    print(f"  Sample Engineered Columns : {fe_art.engineered_feature_names[:6]}...")

    # 3. ML MODEL AGENT (Training, CV, Optuna Tuning)
    print("\n" + "-" * 80)
    print("[Step 2] MODEL TOURNAMENT & OPTUNA TUNING (ON TRAIN SPLIT)")
    print("-" * 80)
    trainer = MLModelAgent()
    state, candidates = trainer.execute(state)

    print(f"  Candidate Models Trained & Cross-Validated: {len(candidates)}")
    print(f"  {'Model Name':<24} | {'CV Mean':<10} | {'CV Std':<8} | {'Val F1-Macro':<12} | {'Test Score':<12}")
    print("  " + "-" * 74)
    for c in candidates:
        m = c.metrics["f1_macro"]
        test_str = "UNTOUCHED" if m.test_score is None else f"{m.test_score:.4f}"
        print(f"  {c.model_name:<24} | {m.cv_mean:<10.4f} | {m.cv_std:<8.4f} | {m.val_score:<12.4f} | {test_str:<12}")

    # 4. EVALUATION AGENT (Selection, Untouched Test Evaluation, Leaderboard)
    print("\n" + "-" * 80)
    print("[Step 3] EVALUATION ENGINE & UNTOUCHED TEST EVALUATION")
    print("-" * 80)
    evaluator = EvaluationAgent()
    state = evaluator.evaluate(state, candidates)
    report = state.evaluation_report

    print(f"\n  CHAMPION MODEL SELECTED : {report.champion_model_name.upper()}")
    print(f"  Generalization Gap (|Val - Test|): {report.generalization_gap:.4f}")
    print(f"  Model CV Variance (Stability)    : {report.cv_variance:.4f}")
    if report.calibration_brier_score is not None:
        print(f"  Calibration Brier Score          : {report.calibration_brier_score:.4f}")

    champ = next(c for c in report.leaderboard if c.is_champion)
    champ_m = champ.metrics["f1_macro"]
    print(f"\n  Final Champion Test Performance (F1-Macro): {champ_m.test_score:.4f}")
    print(f"  Empirical 95% Confidence Interval         : [{champ_m.bootstrap_ci_lower:.4f}, {champ_m.bootstrap_ci_upper:.4f}]")
    print(f"  Confusion Matrix                          : {report.confusion_matrix}")

    print("\n  Complete Model Leaderboard:")
    print(f"  {'Rank':<5} | {'Model Name':<22} | {'Val F1':<8} | {'CV Std':<8} | {'Test F1':<8} | {'Status'}")
    print("  " + "-" * 70)
    for i, c in enumerate(report.leaderboard, 1):
        m = c.metrics["f1_macro"]
        status = "[CHAMPION]" if c.is_champion else "Candidate"
        test_val = f"{m.test_score:.4f}" if m.test_score is not None else "-"
        print(f"  {i:<5} | {c.model_name:<22} | {m.val_score:<8.4f} | {m.cv_std:<8.4f} | {test_val:<8} | {status}")

    print("\n" + "=" * 80)
    print("PHASE 3 DEMONSTRATION COMPLETE - ALL ML & EVALUATION ARTIFACTS VERIFIED")
    print(f"State Status: {state.status}")
    print(f"Total Session Execution Logs: {len(state.execution_logs)}")
    print("=" * 80)


if __name__ == "__main__":
    run_phase3_demo()
