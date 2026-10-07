"""
Demonstration script for Phase 4: Critic Agent and Self-Reflective Closed-Loop Orchestration.

Scenario
────────
1. Generate a credit-default dataset that CONTAINS a deliberately leaky feature
   ('risk_score' ≈ target + tiny noise) so the initial pipeline gets REJECTED.

2. Run the full pipeline (FE → Training → Evaluation).

3. Launch the SelfReflectionOrchestrator.
   Cycle 1 → Critic detects the leaky feature → REJECTED
   Cycle 2 → Orchestrator drops the feature, retrains, re-evaluates → APPROVED

4. Print a structured timeline showing before/after metrics, every cycle's
   verdict, and the final champion.
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
from core.orchestrator import SelfReflectionOrchestrator
from core.config import WORKSPACE_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generator (includes deliberate leaky feature)
# ─────────────────────────────────────────────────────────────────────────────

def generate_leaky_credit_dataset(csv_path: Path, n: int = 400) -> None:
    """
    Generates a credit-default classification dataset with a deliberately leaky
    feature ('risk_score') that is almost identical to the target label.
    This will cause the initial pipeline to be REJECTED by the Critic Agent.
    """
    np.random.seed(99)

    credit_score      = np.random.normal(680, 50, n).clip(300, 850)
    annual_income     = np.random.exponential(55_000, n).clip(15_000, 300_000)
    debt_to_income    = np.random.uniform(5, 55, n)
    employment_type   = np.random.choice(
        ["Salaried", "Self-Employed", "Business", "Unemployed"], n, p=[0.5, 0.25, 0.2, 0.05]
    )
    loan_purpose      = np.random.choice(["Home", "Auto", "Education", "Personal"], n)

    logit = (
        -0.012 * (credit_score - 650)
        + 0.06  * (debt_to_income - 25)
        - 0.00002 * (annual_income - 50_000)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default_target = (np.random.rand(n) < prob_default).astype(int)

    # ── Leaky feature: nearly deterministic copy of the target ──────────────
    # |Pearson corr with target| ≈ 0.97–0.99 → will trigger V1 + V5 CRITICAL_FAIL
    leaky_risk_score = default_target.astype(float) + np.random.normal(0, 0.04, n)

    df = pd.DataFrame({
        "credit_score":    credit_score,
        "annual_income":   annual_income,
        "debt_to_income":  debt_to_income,
        "employment_type": employment_type,
        "loan_purpose":    loan_purpose,
        "risk_score":      leaky_risk_score,   # <<< intentionally leaky
        "default":         default_target,
    })
    df.to_csv(csv_path, index=False)
    print(f"  Generated dataset at '{csv_path}' "
          f"({len(df)} rows, {len(df.columns)} cols, includes LEAKY feature 'risk_score').")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _divider(char: str = "=", width: int = 72) -> str:
    return char * width


def _print_audit_table(feedback) -> None:
    """Prints a compact audit table for one CriticFeedback."""
    icon = {"PASS": "[PASS]", "WARNING": "[WARN]", "CRITICAL_FAIL": "[CRIT]"}
    print(f"  {'V#':<4} {'Check':<36} {'Status':<12} {'Key Evidence'}")
    print("  " + "-" * 70)
    for item in feedback.audit_results:
        ev_str = ""
        if item.evidence:
            # Show at most 2 key-value pairs
            ev_pairs = list(item.evidence.items())[:2]
            ev_str = ", ".join(f"{k}={v}" for k, v in ev_pairs)
        print(f"  V{item.vector_id:<3} {item.check_name:<36} "
              f"{icon.get(item.status, item.status):<12} {ev_str}")


def _print_cycle_summary(entry) -> None:
    """Prints one reflection-cycle log entry."""
    verdict_tag = "[APPROVED]" if entry.verdict == "APPROVED" else "[REJECTED]"
    print(f"\n  Cycle {entry.cycle_number}  {verdict_tag}  Health={entry.health_score:.1f}/100")
    if entry.critical_issues:
        print(f"  Critical issues  : {', '.join(entry.critical_issues)}")
    if entry.warnings_raised:
        print(f"  Warnings         : {', '.join(entry.warnings_raised)}")
    if entry.corrective_action:
        # Truncate long directives for readability
        directive = entry.corrective_action[:120] + "..." \
            if len(entry.corrective_action) > 120 else entry.corrective_action
        print(f"  Directive        : {directive}")
    if entry.agent_rerun:
        print(f"  Agent rerun      : {entry.agent_rerun}")
    if entry.metrics_before:
        print(f"  Metrics (before) : {entry.metrics_before}")
    if entry.metrics_after:
        print(f"  Metrics (after)  : {entry.metrics_after}")
    if entry.stopped_reason:
        print(f"  Stopped because  : {entry.stopped_reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Main demonstration
# ─────────────────────────────────────────────────────────────────────────────

def run_phase4_demo() -> None:
    print(_divider())
    print("AUTONOMOUS AI DATA SCIENTIST - PHASE 4 DEMONSTRATION")
    print("Critic Agent + Self-Reflective Closed-Loop Orchestration")
    print(_divider())

    session_id = "demo_phase4_session"
    raw_csv    = WORKSPACE_DIR / "sample_leaky_credit_data.csv"

    print("\n[Setup] Generating intentionally problematic dataset...")
    generate_leaky_credit_dataset(raw_csv)

    # ── Stage 1: Ingest ────────────────────────────────────────────────────
    ingestor = DatasetIngestor(session_id=session_id)
    ingest   = ingestor.ingest_csv(raw_csv)

    state = DataScienceState(
        session_id=session_id,
        user_goal=(
            "Predict loan default. The dataset contains a suspicious 'risk_score' feature – "
            "the Critic should detect and eliminate it."
        ),
        raw_dataset_path=ingest.stored_path,
    )

    print("\n[Stage 1] Profiling + Cleaning...")
    DataAnalystAgent().execute(state)
    DataCleaningAgent().execute(state)

    # ── Stage 2: Initial pipeline (FE → Training → Evaluation) ───────────
    print("[Stage 2] Feature Engineering + Model Tournament (INITIAL RUN)...")
    FeatureEngineeringAgent().execute(state)
    state, candidates = MLModelAgent().execute(state)
    EvaluationAgent().evaluate(state, candidates)

    initial_report  = state.evaluation_report
    initial_champion = state.champion_model_name
    initial_metrics  = {}
    if initial_report and initial_report.leaderboard:
        champ = next((c for c in initial_report.leaderboard if c.is_champion), None)
        if champ:
            m = champ.metrics.get("f1_macro")
            if m:
                initial_metrics = {
                    "val_score":  m.val_score,
                    "test_score": m.test_score,
                    "cv_std":     m.cv_std,
                }

    print(f"\n  Initial champion : {initial_champion}")
    print(f"  Initial metrics  : {initial_metrics}")

    print("\n" + _divider("-"))
    print("[Stage 3] PRE-REFLECTION CRITIC AUDIT (before orchestrator)")
    print(_divider("-"))
    from agents.critic import CriticAgent
    pre_fb = CriticAgent().audit(state, candidates, iteration=0)
    print(f"\n  Pre-reflection Verdict : {pre_fb.verdict}  "
          f"(Health={pre_fb.health_score:.1f}/100)")
    _print_audit_table(pre_fb)
    if pre_fb.specific_directives:
        print(f"\n  Corrective plan : {pre_fb.specific_directives[:200]}")

    # ── Stage 3: Self-Reflection Orchestration ────────────────────────────
    print("\n" + _divider())
    print("SELF-REFLECTION DEMONSTRATION  (max 3 cycles)")
    print(_divider())

    orch = SelfReflectionOrchestrator(max_reflection_cycles=3)
    final_state, final_candidates = orch.run(state, candidates)

    # ── Print reflection timeline ──────────────────────────────────────────
    print(f"\n  Total reflection cycles executed : {len(final_state.reflection_log)}")
    print()
    for entry in final_state.reflection_log:
        _print_cycle_summary(entry)

    # ── Final audit ────────────────────────────────────────────────────────
    print("\n" + _divider("-"))
    print("FINAL STATE AFTER SELF-REFLECTION")
    print(_divider("-"))

    final_report  = final_state.evaluation_report
    final_champ   = final_state.champion_model_name
    final_metrics = {}
    if final_report and final_report.leaderboard:
        champ = next((c for c in final_report.leaderboard if c.is_champion), None)
        if champ:
            m = champ.metrics.get("f1_macro")
            if m:
                final_metrics = {
                    "val_score":  m.val_score,
                    "test_score": m.test_score,
                    "cv_std":     m.cv_std,
                }

    last_cycle = final_state.reflection_log[-1]

    print(f"\n  Final champion   : {final_champ}")
    print(f"  Final verdict    : {last_cycle.verdict}")
    print(f"  Health score     : {last_cycle.health_score:.1f}/100")
    print(f"  Stopped reason   : {last_cycle.stopped_reason}")

    print(f"\n  Metrics BEFORE correction : {initial_metrics}")
    print(f"  Metrics AFTER  correction : {final_metrics}")

    if (initial_metrics.get("val_score") is not None
            and final_metrics.get("val_score") is not None):
        val_before = initial_metrics["val_score"]
        val_after  = final_metrics["val_score"]
        print(f"\n  Val F1 change : {val_before:.4f} -> {val_after:.4f}  "
              f"({'improved' if val_after < val_before else 'changed'} after leakage removal)")

    print("\n  Critic Audit History:")
    for fb in final_state.critic_history:
        n_crit = sum(1 for a in fb.audit_results if a.status == "CRITICAL_FAIL")
        n_warn = sum(1 for a in fb.audit_results if a.status == "WARNING")
        print(f"    Cycle {fb.iteration}: {fb.verdict}  health={fb.health_score:.1f}  "
              f"critical={n_crit}  warnings={n_warn}")

    print("\n" + _divider())
    print("PHASE 4 DEMONSTRATION COMPLETE")
    print(f"  State status              : {final_state.status}")
    print(f"  Total execution logs      : {len(final_state.execution_logs)}")
    print(f"  Reflection cycles run     : {len(final_state.reflection_log)}")
    print(f"  Critic audit records      : {len(final_state.critic_history)}")
    print(_divider())


if __name__ == "__main__":
    run_phase4_demo()
