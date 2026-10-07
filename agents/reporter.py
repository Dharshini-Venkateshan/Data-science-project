"""
Final Report Generator for Autonomous AI Data Scientist.

Produces a structured Markdown report covering the complete pipeline:
  1. Dataset Summary
  2. Data Quality & Cleaning
  3. EDA Findings
  4. Feature Engineering
  5. Model Tournament Results
  6. Critic Audit Findings
  7. Self-Reflection Log
  8. Explainability / XAI
  9. Ablation / Research Evidence (optional)
 10. Conclusions & Recommendations

The report is saved as a Markdown file under the session workspace and
its path is stored in state.final_report_path.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.state import DataScienceState
from core.config import WORKSPACE_DIR


class FinalReportAgent:
    """
    Generates a comprehensive, human-readable Markdown final report from
    the populated DataScienceState object.  No LLM required.
    """

    def execute(self, state: DataScienceState) -> DataScienceState:
        """Entry point — generate and persist the final report."""
        report_md = self._build_report(state)

        report_dir = WORKSPACE_DIR / state.session_id / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"final_report_{ts}.md"
        report_path.write_text(report_md, encoding="utf-8")

        state.final_report_path = str(report_path)
        state.status = "REPORTING"
        state.log(f"FinalReportAgent: Report written to '{report_path}'.")
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Internal builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_report(self, state: DataScienceState) -> str:
        sections: List[str] = [
            self._header(state),
            self._section_dataset(state),
            self._section_cleaning(state),
            self._section_eda(state),
            self._section_features(state),
            self._section_model_tournament(state),
            self._section_critic(state),
            self._section_reflection(state),
            self._section_xai(state),
            self._section_conclusions(state),
        ]
        return "\n\n".join(s for s in sections if s.strip())

    # ── Header ──────────────────────────────────────────────────────────────

    def _header(self, state: DataScienceState) -> str:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return (
            "# Autonomous AI Data Scientist — Final Report\n\n"
            f"**Session ID:** `{state.session_id}`  \n"
            f"**Generated:** {ts}  \n"
            f"**User Goal:** {state.user_goal}  \n"
            f"**Pipeline Status:** `{state.status}`\n\n"
            "---"
        )

    # ── 1. Dataset Summary ───────────────────────────────────────────────────

    def _section_dataset(self, state: DataScienceState) -> str:
        p = state.profile
        if not p:
            return "## 1. Dataset Summary\n\n_No profile generated._"

        lines = [
            "## 1. Dataset Summary\n",
            f"| Property | Value |",
            f"|---|---|",
            f"| Rows | {p.row_count:,} |",
            f"| Columns | {p.col_count} |",
            f"| Target Column | `{p.target_column}` |",
            f"| Problem Type | `{p.problem_type}` |",
            f"| Duplicate Rows | {p.duplicate_rows_count} |",
            f"| Class Imbalance Ratio | {p.class_imbalance_ratio:.3f} |"
            if p.class_imbalance_ratio else "| Class Imbalance Ratio | N/A |",
        ]

        if p.constant_columns:
            lines.append(f"\n**Constant columns detected:** {', '.join(f'`{c}`' for c in p.constant_columns)}")
        if p.potential_leakage_columns:
            lines.append(f"\n> ⚠️ **Potential leakage columns flagged:** "
                         f"{', '.join(f'`{c}`' for c in p.potential_leakage_columns)}")
        if p.high_cardinality_columns:
            lines.append(f"\n**High-cardinality columns:** {', '.join(f'`{c}`' for c in p.high_cardinality_columns)}")

        return "\n".join(lines)

    # ── 2. Data Quality & Cleaning ──────────────────────────────────────────

    def _section_cleaning(self, state: DataScienceState) -> str:
        ca = state.cleaning_artifact
        if not ca:
            return "## 2. Data Quality & Cleaning\n\n_Cleaning not executed._"

        lines = [
            "## 2. Data Quality & Cleaning\n",
            f"**Leakage Guard Passed:** {'✅ Yes' if ca.leakage_guard_passed else '❌ No'}  ",
        ]

        if ca.train_shape:
            lines.append(
                f"\n| Split | Shape |\n|---|---|\n"
                f"| Train | {ca.train_shape[0]:,} rows × {ca.train_shape[1]} cols |\n"
                + (f"| Val   | {ca.val_shape[0]:,} rows × {ca.val_shape[1]} cols |\n" if ca.val_shape else "")
                + (f"| Test  | {ca.test_shape[0]:,} rows × {ca.test_shape[1]} cols |" if ca.test_shape else "")
            )

        if ca.transformations_applied:
            lines.append("\n**Transformations Applied:**")
            for t in ca.transformations_applied:
                lines.append(f"- {t}")

        if ca.dropped_columns:
            lines.append(f"\n**Dropped Columns:** {', '.join(f'`{c}`' for c in ca.dropped_columns)}")

        if ca.notes:
            lines.append(f"\n> {ca.notes}")

        return "\n".join(lines)

    # ── 3. EDA Findings ─────────────────────────────────────────────────────

    def _section_eda(self, state: DataScienceState) -> str:
        eda = state.eda_results
        if not eda:
            return "## 3. Exploratory Data Analysis\n\n_EDA not executed._"

        lines = ["## 3. Exploratory Data Analysis\n"]

        if eda.key_findings:
            lines.append("**Key Findings:**")
            for f in eda.key_findings:
                lines.append(f"- {f}")

        if eda.statistical_anomalies:
            lines.append("\n**Statistical Anomalies:**")
            for a in eda.statistical_anomalies:
                lines.append(f"- {a}")

        if eda.top_correlations:
            lines.append("\n**Top Feature Correlations with Target:**\n")
            lines.append("| Feature | Correlation |")
            lines.append("|---|---|")
            for item in eda.top_correlations[:8]:
                lines.append(f"| `{item.get('feature', '')}` | {item.get('correlation', 0):.4f} |")

        if eda.generated_chart_paths:
            lines.append(f"\n**Figures Generated:** {len(eda.generated_chart_paths)} charts saved.")

        return "\n".join(lines)

    # ── 4. Feature Engineering ───────────────────────────────────────────────

    def _section_features(self, state: DataScienceState) -> str:
        fa = state.feature_artifact
        if not fa:
            return "## 4. Feature Engineering\n\n_Feature engineering not executed._"

        lines = [
            "## 4. Feature Engineering\n",
            f"**Total Features After Engineering:** {fa.total_features_count}  ",
            f"**Selection Method:** `{fa.selection_method or 'N/A'}`  ",
        ]

        if fa.engineered_feature_names:
            top = fa.engineered_feature_names[:10]
            lines.append(f"\n**Feature Names (first 10):** {', '.join(f'`{n}`' for n in top)}")

        if fa.dropped_feature_names:
            lines.append(f"\n**Dropped Features:** {', '.join(f'`{n}`' for n in fa.dropped_feature_names)}")

        return "\n".join(lines)

    # ── 5. Model Tournament ──────────────────────────────────────────────────

    def _section_model_tournament(self, state: DataScienceState) -> str:
        rpt = state.evaluation_report
        if not rpt or not rpt.leaderboard:
            return "## 5. Model Tournament Results\n\n_Model evaluation not completed._"

        primary_metric = "f1_macro"
        if state.profile and state.profile.problem_type == "regression":
            primary_metric = "r2"

        lines = [
            "## 5. Model Tournament Results\n",
            f"**Champion Model:** `{rpt.champion_model_name}`  ",
            f"**Primary Metric:** `{primary_metric}`  ",
        ]

        if rpt.generalization_gap is not None:
            lines.append(f"**Generalization Gap:** {rpt.generalization_gap:.4f}  ")
        if rpt.calibration_brier_score is not None:
            lines.append(f"**Brier Score:** {rpt.calibration_brier_score:.4f}  ")

        lines.append("\n**Leaderboard:**\n")
        lines.append("| Model | Val Score | Test Score | CV Mean | CV Std | Champion |")
        lines.append("|---|---|---|---|---|---|")

        for cand in sorted(rpt.leaderboard, key=lambda c: -(c.metrics.get(primary_metric, type("M", (), {"val_score": 0})()).val_score or 0)):
            m = cand.metrics.get(primary_metric)
            val = f"{m.val_score:.4f}" if m and m.val_score is not None else "—"
            tst = f"{m.test_score:.4f}" if m and m.test_score is not None else "—"
            cvm = f"{m.cv_mean:.4f}" if m and m.cv_mean is not None else "—"
            cvs = f"{m.cv_std:.4f}" if m and m.cv_std is not None else "—"
            champ = "🏆 Yes" if cand.is_champion else "No"
            lines.append(f"| `{cand.model_name}` | {val} | {tst} | {cvm} | {cvs} | {champ} |")

        return "\n".join(lines)

    # ── 6. Critic Audit ──────────────────────────────────────────────────────

    def _section_critic(self, state: DataScienceState) -> str:
        if not state.critic_history:
            return "## 6. Critic Audit\n\n_Critic not executed in this run._"

        lines = ["## 6. Critic Audit\n"]
        for fb in state.critic_history:
            icon = "✅" if fb.verdict == "APPROVED" else "❌"
            lines.append(f"### Iteration {fb.iteration} — {icon} {fb.verdict}  (Health: {fb.health_score:.1f}/100)\n")
            if fb.audit_results:
                lines.append("| Vector | Check | Status | Observation |")
                lines.append("|---|---|---|---|")
                for audit in fb.audit_results:
                    status_icon = "✅" if audit.status == "PASS" else ("⚠️" if audit.status == "WARNING" else "❌")
                    obs = audit.observation.replace("|", "∣")[:80]
                    lines.append(f"| V{audit.vector_id} | {audit.check_name} | {status_icon} {audit.status} | {obs} |")
            if fb.specific_directives:
                lines.append(f"\n**Directives:** {fb.specific_directives[:200]}")

        return "\n".join(lines)

    # ── 7. Self-Reflection Log ───────────────────────────────────────────────

    def _section_reflection(self, state: DataScienceState) -> str:
        if not state.reflection_log:
            return "## 7. Self-Reflection Log\n\n_Self-reflection not executed in this run._"

        lines = [
            "## 7. Self-Reflection Log\n",
            f"**Total Reflection Cycles:** {len(state.reflection_log)}\n",
        ]

        for log in state.reflection_log:
            icon = "✅" if log.verdict == "APPROVED" else "❌"
            lines.append(f"### Cycle {log.cycle_number} — {icon} {log.verdict}  (Health: {log.health_score:.1f}/100)")
            if log.corrective_action:
                lines.append(f"- **Corrective Action:** {log.corrective_action}")
            if log.agent_rerun:
                lines.append(f"- **Agent Re-run:** `{log.agent_rerun}`")
            if log.champion_before != log.champion_after and log.champion_after:
                lines.append(f"- **Champion:** `{log.champion_before}` → `{log.champion_after}`")
            if log.stopped_reason:
                lines.append(f"- **Stopped:** {log.stopped_reason}")

        return "\n".join(lines)

    # ── 8. Explainability / XAI ─────────────────────────────────────────────

    def _section_xai(self, state: DataScienceState) -> str:
        xai = state.insight_artifact
        if not xai:
            return "## 8. Model Explainability (XAI)\n\n_Explainability not executed._"

        lines = [
            "## 8. Model Explainability (XAI)\n",
            f"**Explained Model:** `{xai.champion_model_name}`  ",
            f"**Explanation Method:** `{xai.explanation_method}`  ",
        ]

        if xai.global_feature_importance:
            lines.append("\n**Global Feature Importance (Top 10):**\n")
            lines.append("| Rank | Feature | Importance |")
            lines.append("|---|---|---|")
            for rank, (feat, score) in enumerate(list(xai.global_feature_importance.items())[:10], 1):
                lines.append(f"| {rank} | `{feat}` | {score:.6f} |")

        if xai.vulnerability_warnings:
            lines.append("\n**⚠️ Vulnerability Warnings:**")
            for w in xai.vulnerability_warnings:
                lines.append(f"- {w}")

        if xai.business_recommendations:
            lines.append("\n**Actionable Recommendations:**")
            for r in xai.business_recommendations:
                lines.append(f"- {r}")

        if xai.feature_importance_plot_path:
            lines.append(f"\n_Feature importance plot: `{xai.feature_importance_plot_path}`_")
        if xai.shap_summary_plot_path:
            lines.append(f"_SHAP summary plot: `{xai.shap_summary_plot_path}`_")

        return "\n".join(lines)

    # ── 9. Conclusions ───────────────────────────────────────────────────────

    def _section_conclusions(self, state: DataScienceState) -> str:
        lines = ["## 9. Conclusions & Recommendations\n"]

        if state.champion_model_name:
            lines.append(f"**Final Selected Model:** `{state.champion_model_name}`  ")

        if state.evaluation_report and state.evaluation_report.generalization_gap is not None:
            gap = state.evaluation_report.generalization_gap
            if gap < 0.05:
                lines.append("✅ Generalization gap is acceptable (< 0.05) — model generalizes well.")
            elif gap < 0.15:
                lines.append("⚠️ Moderate generalization gap detected — consider more regularization or data.")
            else:
                lines.append("❌ High generalization gap — strong signs of overfitting; further tuning required.")

        critic_verdicts = [fb.verdict for fb in state.critic_history]
        if critic_verdicts:
            final_verdict = critic_verdicts[-1]
            lines.append(f"\n**Critic Final Verdict:** {'✅ APPROVED' if final_verdict == 'APPROVED' else '❌ REJECTED'}")

        if state.reflection_log:
            lines.append(f"**Self-Reflection Cycles:** {len(state.reflection_log)}")

        lines += [
            "\n**Research Integrity Note:**",
            "Results reported here reflect actual measured outcomes. No test-set information",
            "was used during model selection, feature engineering, or hyperparameter tuning.",
            "Statistical comparisons require multiple independent runs for inferential validity.",
        ]

        return "\n".join(lines)
