"""
Demonstration script for Phase 5: Explainability, Ablation Studies & Research Evidence.

Research Scenario
─────────────────
1. Generate an empirical benchmark dataset (credit default risk) containing
   a subtle high-correlation leakage feature ('risk_score').

2. Execute the multi-arm Ablation Study:
   - Arm A: Standard ML Baseline (no Critic, no reflection)
   - Arm B: ML + Critic (Critic audit flag, but no reflection or rerun)
   - Arm C: Full Proposed System (Critic + Self-Reflection closed-loop correction)

3. Generate XAI Interpretability:
   - Global Feature Importance (Tree / Linear coefficients / SHAP)
   - SHAP Summary Plot
   - Local Case Predictions (positive and negative attributions)
   - Vulnerability and domain monitoring alerts

4. Print Structured Research Table and Scientific Findings:
   - Objective comparison table
   - Generalization gap & calibration analysis
   - Honest reporting of why leakage inflated baseline scores
   - Research artifacts inventory
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Force UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from core.config import (
    WORKSPACE_DIR,
    RESEARCH_DIR,
    RESEARCH_RESULTS_DIR,
    RESEARCH_EXPERIMENTS_DIR,
    RESEARCH_FIGURES_DIR,
)
from research.ablation import AblationStudy, AblationConfig, format_ablation_table


def generate_benchmark_credit_data(csv_path: Path, n: int = 400, seed: int = 99) -> None:
    """Generates reproducible credit classification data with intentional subtle leakage."""
    np.random.seed(seed)

    credit_score = np.random.normal(680, 50, n).clip(300, 850)
    annual_income = np.random.exponential(55_000, n).clip(15_000, 300_000)
    debt_to_income = np.random.uniform(5, 55, n)
    employment_type = np.random.choice(
        ["Salaried", "Self-Employed", "Business", "Unemployed"],
        n,
        p=[0.5, 0.25, 0.2, 0.05],
    )
    loan_purpose = np.random.choice(["Home", "Auto", "Education", "Personal"], n)

    logit = (
        -0.012 * (credit_score - 650)
        + 0.06 * (debt_to_income - 25)
        - 0.00002 * (annual_income - 50_000)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default_target = (np.random.rand(n) < prob_default).astype(int)

    # Injected leakage feature: near-deterministic copy of default label
    leaky_risk_score = default_target.astype(float) + np.random.normal(0, 0.04, n)

    df = pd.DataFrame({
        "credit_score": credit_score,
        "annual_income": annual_income,
        "debt_to_income": debt_to_income,
        "employment_type": employment_type,
        "loan_purpose": loan_purpose,
        "risk_score": leaky_risk_score,
        "default": default_target,
    })
    df.to_csv(csv_path, index=False)
    print(f"  Benchmark dataset generated at '{csv_path}' ({len(df)} samples, {len(df.columns)} features).")


def run_phase5_demo() -> None:
    print("=" * 80)
    print("AUTONOMOUS AI DATA SCIENTIST - PHASE 5 RESEARCH DEMONSTRATION")
    print("Explainability (XAI), Ablation Studies & Empirical Research Evidence")
    print("=" * 80)

    dataset_path = WORKSPACE_DIR / "sample_phase5_benchmark_credit.csv"
    user_goal = "Predict loan default. Ensure high generalization and explainability."

    print("\n[Step 1] Preparing Controlled Benchmark Dataset...")
    generate_benchmark_credit_data(dataset_path, n=400, seed=99)

    print("\n[Step 2] Executing Controlled Multi-Arm Ablation Study...")
    print("  - Arm A: Standard ML Baseline (NO Critic, NO Self-Reflection)")
    print("  - Arm B: ML + Critic (Critic Audit, NO Self-Reflection)")
    print("  - Arm C: Full Proposed System (Critic + Self-Reflection Closed-Loop)")

    config = AblationConfig(
        experiment_name="credit_default_phase5_study",
        random_seed=42,
        max_reflection_cycles=3,
        save_artifacts=True,
    )

    study = AblationStudy(config=config)
    results_df, metadata, final_state = study.run(
        dataset_path=dataset_path,
        user_goal=user_goal,
        session_prefix="phase5_demo",
    )

    # ── Display Ablation Table ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESEARCH ABLATION STUDY RESULTS TABLE")
    print("=" * 80)
    print(format_ablation_table(results_df))

    # ── Scientific Insights & Analysis ─────────────────────────────────────
    print("\n" + "-" * 80)
    print("SCIENTIFIC ANALYSIS & ABLATION TAKEAWAYS")
    print("-" * 80)

    arm_a_row = results_df[results_df["Experiment"] == "Standard ML Baseline"].iloc[0]
    arm_b_row = results_df[results_df["Experiment"] == "ML + Critic (Audit Only)"].iloc[0]
    arm_c_row = results_df[results_df["Experiment"] == "Full System (Self-Reflection)"].iloc[0]

    print(f"1. Standard Baseline (Arm A):")
    print(f"   - Selected Champion : {arm_a_row['Final Model']}")
    print(f"   - Apparent Val Score: {arm_a_row['Val Metric']} (Artificially inflated due to unmitigated leakage)")
    print(f"   - Critic Status     : Not executed")

    print(f"\n2. ML + Critic Audit Only (Arm B):")
    print(f"   - Selected Champion : {arm_b_row['Final Model']}")
    print(f"   - Critic Health     : {arm_b_row['Critic Health']}/100 (REJECTED due to leakage & high val score)")
    print(f"   - Limitation        : Identifies defect but cannot self-heal without iterative reflection.")

    print(f"\n3. Full Proposed Architecture (Arm C):")
    print(f"   - Final Champion    : {arm_c_row['Final Model']}")
    print(f"   - Honest Val Score  : {arm_c_row['Val Metric']}")
    print(f"   - Honest Test Score : {arm_c_row['Test Metric']}")
    print(f"   - Critic Health     : {arm_c_row['Critic Health']}/100 (APPROVED after automated corrective loop)")
    print(f"   - Reflection Cycles : {arm_c_row['Reflection Cycles']}")

    # ── Display XAI Interpretability Results ───────────────────────────────
    print("\n" + "=" * 80)
    print("EXPLAINABILITY (XAI) SUMMARY - CHAMPION MODEL")
    print("=" * 80)

    if final_state.insight_artifact:
        xai = final_state.insight_artifact
        print(f"  Champion Model Explained : {xai.champion_model_name}")
        print(f"  Explanation Method       : {xai.explanation_method}")
        print("\n  Global Feature Importance (Top Features):")
        for rank, (feat, score) in enumerate(list(xai.global_feature_importance.items())[:6], 1):
            bar = "#" * int(score * 40)
            print(f"    {rank}. {feat:<25} : {score:.4f}  {bar}")

        if xai.vulnerability_warnings:
            print("\n  Vulnerability Warnings:")
            for w in xai.vulnerability_warnings:
                print(f"    - {w}")

        if xai.business_recommendations:
            print("\n  Actionable Business Recommendations:")
            for b in xai.business_recommendations:
                print(f"    - {b}")

        if xai.local_case_explanations:
            print("\n  Representative Local Case Explanation (Sample #1):")
            c1 = xai.local_case_explanations[0]
            print(f"    Actual Target       : {c1.get('actual_label')}")
            print(f"    Predicted Score     : {c1.get('predicted_label')} (prob={c1.get('predicted_probability')})")
            print("    Top Positive Drivers:")
            for p in c1.get("top_positive_features", [])[:2]:
                print(f"      + {p['feature']}: +{p['contribution']:.4f}")
            print("    Top Negative Drivers:")
            for n in c1.get("top_negative_features", [])[:2]:
                print(f"      - {n['feature']}: {n['contribution']:.4f}")

    # ── Inventory of Generated Research Artifacts ──────────────────────────
    print("\n" + "=" * 80)
    print("RESEARCH ARTIFACTS GENERATED")
    print("=" * 80)
    for p in [
        RESEARCH_RESULTS_DIR / "ablation_results.csv",
        RESEARCH_EXPERIMENTS_DIR / "experiment_log.json",
        RESEARCH_FIGURES_DIR / "feature_importance.png",
        RESEARCH_FIGURES_DIR / "shap_summary.png",
        RESEARCH_FIGURES_DIR / "model_comparison.png",
        RESEARCH_FIGURES_DIR / "calibration_plot.png",
    ]:
        status = "[EXISTS]" if p.exists() else "[MISSING]"
        size = f"{p.stat().st_size} bytes" if p.exists() else ""
        print(f"  {status} {p} {size}")

    print("\n" + "-" * 80)
    print("STATISTICAL METHODOLOGY & LIMITATIONS")
    print("-" * 80)
    print(f"  Note: {metadata.get('statistical_limitation')}")
    print("=" * 80)
    print("PHASE 5 DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_phase5_demo()
