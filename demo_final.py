"""
demo_final.py -- Complete End-to-End Demonstration
Autonomous AI Data Scientist

Runs the full pipeline:
  DATA INGESTION -> PROFILING -> CLEANING -> EDA ->
  FEATURE ENGINEERING -> MODEL TOURNAMENT -> EVALUATION ->
  CRITIC AUDIT -> SELF-REFLECTION -> EXPLAINABILITY -> FINAL REPORT

Usage:
    python demo_final.py [--csv PATH] [--goal "GOAL TEXT"]

Default dataset: workspace/sample_credit_data.csv
"""

import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows so Unicode characters print correctly.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 — ignore

# -- Project imports -----------------------------------------------------------
from core.config import WORKSPACE_DIR
from core.state import DataScienceState
from tools.ingestion import DatasetIngestor
from agents.analyst import DataAnalystAgent
from agents.cleaner import DataCleaningAgent
from agents.eda import EDAAgent
from agents.feature_engineer import FeatureEngineeringAgent
from agents.model_trainer import MLModelAgent
from agents.evaluator import EvaluationAgent
from agents.critic import CriticAgent
from core.orchestrator import SelfReflectionOrchestrator
from agents.xai import XAIAgent
from agents.reporter import FinalReportAgent


# -- Helpers -------------------------------------------------------------------

def _banner(title: str, width: int = 70) -> str:
    pad = (width - len(title) - 2) // 2
    return f"\n{'=' * width}\n{' ' * pad} {title}\n{'=' * width}"


def _step(step_num: int, description: str) -> None:
    print(f"\n{'-' * 70}")
    print(f"  STEP {step_num:02d} | {description}")
    print(f"{'-' * 70}")


def _ok(msg: str) -> str:
    return f"  [OK] {msg}"


def _warn(msg: str) -> str:
    return f"  [WARN] {msg}"


# -- Main pipeline -------------------------------------------------------------

def run_full_pipeline(
    csv_path: Path,
    user_goal: str,
    session_id: str = "demo_final_session",
    max_reflection_cycles: int = 3,
) -> DataScienceState:
    print(_banner("AUTONOMOUS AI DATA SCIENTIST -- FULL PIPELINE"))
    print(f"\n  Dataset : {csv_path}")
    print(f"  Goal    : {user_goal}")
    print(f"  Session : {session_id}")

    t_total = time.time()

    # -- Step 1: Data Ingestion ------------------------------------------------
    _step(1, "Data Ingestion")
    ingestor = DatasetIngestor(session_id=session_id)
    ingest = ingestor.ingest_csv(csv_path)
    print(_ok(f"Stored at: {ingest.stored_path}"))

    state = DataScienceState(
        session_id=session_id,
        user_goal=user_goal,
        raw_dataset_path=ingest.stored_path,
    )

    # -- Step 2: Data Profiling ------------------------------------------------
    _step(2, "Data Profiling & Analysis")
    DataAnalystAgent().execute(state)
    p = state.profile
    print(_ok(f"Rows: {p.row_count:,}  |  Cols: {p.col_count}  |  Target: '{p.target_column}'"))
    print(_ok(f"Problem type: {p.problem_type}"))
    if p.potential_leakage_columns:
        print(_warn(f"Potential leakage columns detected: {p.potential_leakage_columns}"))

    # -- Step 3: Data Cleaning -------------------------------------------------
    _step(3, "Data Cleaning & Train/Val/Test Split")
    DataCleaningAgent().execute(state)
    ca = state.cleaning_artifact
    print(_ok(f"Train {ca.train_shape}  |  Val {ca.val_shape}  |  Test {ca.test_shape}"))
    print(_ok(f"Leakage guard: {'PASSED' if ca.leakage_guard_passed else 'FAILED'}"))
    if ca.transformations_applied:
        print(_ok(f"Transformations applied: {len(ca.transformations_applied)}"))

    # -- Step 4: EDA -----------------------------------------------------------
    _step(4, "Exploratory Data Analysis")
    try:
        EDAAgent().execute(state)
    except Exception as e:
        print(_warn(f"EDA partially skipped: {e}"))
    eda = state.eda_results
    if eda:
        print(_ok(f"Key findings: {len(eda.key_findings)}"))
        print(_ok(f"Charts generated: {len(eda.generated_chart_paths)}"))
        if eda.key_findings:
            print(f"  -> {eda.key_findings[0]}")

    # -- Step 5: Feature Engineering -------------------------------------------
    _step(5, "Feature Engineering")
    FeatureEngineeringAgent().execute(state)
    fa = state.feature_artifact
    print(_ok(f"Total features after engineering: {fa.total_features_count}"))

    # -- Step 6: Model Tournament ----------------------------------------------
    _step(6, "Model Tournament & Training")
    state, candidates = MLModelAgent().execute(state)
    print(_ok(f"Trained {len(candidates)} model candidates"))

    # -- Step 7: Evaluation ----------------------------------------------------
    _step(7, "Model Evaluation & Selection")
    EvaluationAgent().evaluate(state, candidates)
    rpt = state.evaluation_report
    print(_ok(f"Champion: {rpt.champion_model_name}"))
    if rpt.generalization_gap is not None:
        print(_ok(f"Generalization gap: {rpt.generalization_gap:.4f}"))
    if rpt.calibration_brier_score is not None:
        print(_ok(f"Brier score: {rpt.calibration_brier_score:.4f}"))

    # -- Step 8: Critic Audit + Self-Reflection --------------------------------
    _step(8, "Critic Audit & Self-Reflection")
    orchestrator = SelfReflectionOrchestrator(max_reflection_cycles=max_reflection_cycles)
    state, candidates = orchestrator.run(state, candidates)

    last_fb = state.critic_history[-1] if state.critic_history else None
    if last_fb:
        print(_ok(f"Critic Final Verdict: {last_fb.verdict}"))
        print(_ok(f"Health Score: {last_fb.health_score:.1f}/100"))
    print(_ok(f"Reflection cycles completed: {len(state.reflection_log)}"))

    # -- Step 9: Explainability ------------------------------------------------
    _step(9, "Model Explainability (XAI)")
    try:
        xai_agent = XAIAgent()
        state = xai_agent.explain(state, candidates=candidates, save_to_research=True)
        xai = state.insight_artifact
        print(_ok(f"Method: {xai.explanation_method}"))
        print(_ok(f"Top feature: {list(xai.global_feature_importance.keys())[0]}"))
        print(_ok(f"Feature importance plot: {xai.feature_importance_plot_path}"))
        if xai.vulnerability_warnings:
            for w in xai.vulnerability_warnings:
                print(_warn(w))
    except Exception as e:
        print(_warn(f"XAI skipped: {e}"))

    # -- Step 10: Final Report -------------------------------------------------
    _step(10, "Final Report Generation")
    FinalReportAgent().execute(state)
    print(_ok(f"Report saved: {state.final_report_path}"))

    elapsed = time.time() - t_total

    # -- Summary ---------------------------------------------------------------
    print(_banner("PIPELINE COMPLETE"))
    print(f"\n  Runtime         : {elapsed:.1f}s")
    print(f"  Champion Model  : {state.champion_model_name}")

    if state.evaluation_report:
        rpt2 = state.evaluation_report
        primary = "f1_macro" if (state.profile and "classification" in state.profile.problem_type) else "r2"
        champ = next((c for c in rpt2.leaderboard if c.is_champion), None)
        if champ:
            m = champ.metrics.get(primary)
            if m:
                print(f"  Val {primary:<12} : {m.val_score:.4f}")
                print(f"  Test {primary:<11} : {m.test_score:.4f}")
                if m.cv_mean is not None:
                    print(f"  CV mean          : {m.cv_mean:.4f} +/- {m.cv_std:.4f}")

    if last_fb:
        print(f"  Critic Health   : {last_fb.health_score:.1f}/100  [{last_fb.verdict}]")
    print(f"  Reflection Cycles: {len(state.reflection_log)}")
    if state.final_report_path:
        print(f"  Final Report    : {state.final_report_path}")

    print("\n" + "=" * 70)
    return state


# -- CLI Entry Point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous AI Data Scientist -- Full Pipeline Demo"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(WORKSPACE_DIR / "sample_credit_data.csv"),
        help="Path to the input CSV dataset",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Predict the probability of credit default. Maximize F1 macro and ensure explainability.",
        help="Natural language description of the ML objective",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="demo_final_session",
        help="Unique session ID for this run",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Maximum self-reflection cycles",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: Dataset not found at '{csv_path}'.")
        sys.exit(1)

    run_full_pipeline(
        csv_path=csv_path,
        user_goal=args.goal,
        session_id=args.session,
        max_reflection_cycles=args.cycles,
    )


if __name__ == "__main__":
    main()
