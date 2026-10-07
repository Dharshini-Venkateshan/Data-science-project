"""
app.py — Streamlit Web UI for Autonomous AI Data Scientist

Run:
    streamlit run app.py
"""

import uuid
from pathlib import Path

import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Autonomous AI Data Scientist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _status_badge(label: str, done: bool, running: bool = False) -> str:
    if done:
        return f"✅ **{label}**"

    if running:
        return f"🔄 **{label}**"

    return f"⏳ {label}"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/000000/robot-2.png",
        width=80,
    )

    st.title("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Target column selection
    # ─────────────────────────────────────────────────────────────────────────

    target_column = None

    if uploaded_file is not None:

        try:

            uploaded_file.seek(0)

            preview_df = pd.read_csv(uploaded_file)

            st.success(
                f"Dataset loaded: {preview_df.shape[0]} rows × "
                f"{preview_df.shape[1]} columns"
            )

            target_column = st.selectbox(
                "🎯 Target Column",
                options=list(preview_df.columns),
                help="Select the column that the ML model should predict.",
            )

            st.caption(
                f"Selected target: **{target_column}**"
            )

        except Exception as e:

            st.error(
                f"Could not read the CSV file: {e}"
            )

    user_goal = st.text_area(
        "🎯 ML Objective",
        placeholder=(
            "Example: Predict employee performance. "
            "Maximize F1 macro."
        ),
        height=90,
    )

    max_cycles = st.slider(
        "🔄 Max Reflection Cycles",
        min_value=1,
        max_value=5,
        value=3,
    )

    run_ablation = st.checkbox(
        "🧪 Also run Ablation Study (slower)",
        value=False,
    )

    run_btn = st.button(
        "▶ Run Full Pipeline",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")

    st.markdown("**Demo datasets**")

    if st.button(
        "Use Credit Risk dataset",
        use_container_width=True,
    ):

        demo_path = WORKSPACE_DIR / "sample_credit_data.csv"

        if demo_path.exists():

            st.session_state["demo_csv"] = str(demo_path)

            st.session_state["demo_goal"] = (
                "Predict credit default. Maximize F1 macro."
            )

            st.rerun()

    if st.button(
        "Use Telecom Churn dataset",
        use_container_width=True,
    ):

        demo_path = WORKSPACE_DIR / "sample_telecom_data.csv"

        if demo_path.exists():

            st.session_state["demo_csv"] = str(demo_path)

            st.session_state["demo_goal"] = (
                "Predict customer churn. Maximize F1 macro."
            )

            st.rerun()

    st.markdown("---")

    st.caption(
        "Autonomous AI Data Scientist v1.0 · Research Prototype"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

st.title("🤖 Autonomous AI Data Scientist")

st.markdown(
    "Upload a CSV, select the target column, describe your ML objective, "
    "and let the multi-agent system handle the full data science pipeline."
)


# ─────────────────────────────────────────────────────────────────────────────
# Demo notification
# ─────────────────────────────────────────────────────────────────────────────

if (
    "demo_csv" in st.session_state
    and "state" not in st.session_state
):

    demo_path = Path(
        st.session_state["demo_csv"]
    )

    if demo_path.exists():

        st.info(
            f"Demo dataset pre-loaded: **`{demo_path.name}`**. "
            "Fill in your ML objective above and click "
            "**▶ Run Full Pipeline**."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────

if run_btn or "state" in st.session_state:

    # ─────────────────────────────────────────────────────────────────────────
    # Resolve input source
    # ─────────────────────────────────────────────────────────────────────────

    csv_path = None

    if uploaded_file is not None:

        tmp_dir = WORKSPACE_DIR / "ui_uploads"

        tmp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        csv_path = tmp_dir / uploaded_file.name

        # Important:
        # reset uploaded-file pointer before reading it
        uploaded_file.seek(0)

        csv_path.write_bytes(
            uploaded_file.read()
        )

    elif "demo_csv" in st.session_state:

        csv_path = Path(
            st.session_state["demo_csv"]
        )

    elif not run_btn and "state" in st.session_state:

        pass

    else:

        st.warning(
            "Please upload a CSV file or select a demo dataset."
        )

        st.stop()


    # ─────────────────────────────────────────────────────────────────────────
    # Run pipeline
    # ─────────────────────────────────────────────────────────────────────────

    if run_btn and csv_path:

        # Target column validation
        if uploaded_file is not None and not target_column:

            st.error(
                "Please select a target column before running the pipeline."
            )

            st.stop()


        session_id = (
            f"ui_session_{uuid.uuid4().hex[:8]}"
        )


        # ─────────────────────────────────────────────────────────────────────
        # Objective
        # ─────────────────────────────────────────────────────────────────────

        goal = user_goal.strip()

        if uploaded_file is not None:

            if not goal:

                goal = (
                    f"Predict {target_column}. "
                    "Maximize predictive performance."
                )

            # Explicitly attach target to objective.
            goal = (
                f"{goal} "
                f"Target column: {target_column}"
            )

        else:

            if not goal:

                goal = st.session_state.get(
                    "demo_goal",
                    "Predict the target column. "
                    "Maximize predictive performance.",
                )


        # ─────────────────────────────────────────────────────────────────────
        # Pipeline steps
        # ─────────────────────────────────────────────────────────────────────

        steps = [
            "Data Ingestion",
            "Data Profiling",
            "Data Cleaning",
            "Exploratory Data Analysis",
            "Feature Engineering",
            "Model Training",
            "Model Evaluation",
            "Critic + Self-Reflection",
            "Explainability (XAI)",
            "Final Report",
        ]


        progress_col, _ = st.columns(
            [3, 1]
        )

        with progress_col:

            progress_bar = st.progress(
                0,
                text="Initializing...",
            )


        step_container = st.container()


        def _update_progress(
            step_i: int,
            message: str,
        ):

            pct = int(
                (step_i / len(steps)) * 100
            )

            progress_bar.progress(
                pct,
                text=message,
            )


        _update_progress(
            0,
            "Starting pipeline...",
        )


        # ─────────────────────────────────────────────────────────────────────
        # Step 1 — Data Ingestion
        # ─────────────────────────────────────────────────────────────────────

        with step_container:

            st.markdown(
                _status_badge(
                    "1. Data Ingestion",
                    done=False,
                    running=True,
                )
            )


        ingestor = DatasetIngestor(
            session_id=session_id
        )

        ingest = ingestor.ingest_csv(
            csv_path
        )


        state = DataScienceState(
            session_id=session_id,
            user_goal=goal,
            raw_dataset_path=ingest.stored_path,
        )


        _update_progress(
            1,
            "Profiling data...",
        )


        # ─────────────────────────────────────────────────────────────────────
        # Step 2 — Profiling
        # ─────────────────────────────────────────────────────────────────────

        DataAnalystAgent().execute(
            state
        )


        # ─────────────────────────────────────────────────────────────────────
        # IMPORTANT TARGET OVERRIDE
        #
        # If the user selected a target column, make sure the state/profile
        # uses that target before cleaning / feature engineering.
        # ─────────────────────────────────────────────────────────────────────

        if uploaded_file is not None and target_column:

            if state.profile is not None:

                state.profile.target_column = target_column

            # Keep target information explicitly available in the state
            # through the user goal as a fallback for downstream agents.
            state.user_goal = (
                f"{goal} Target column: {target_column}"
            )


        _update_progress(
            2,
            "Cleaning data...",
        )


        # ─────────────────────────────────────────────────────────────────────
        # Step 3 — Cleaning
        # ─────────────────────────────────────────────────────────────────────

        DataCleaningAgent().execute(
            state
        )


        _update_progress(
            3,
            "Running EDA...",
        )


        # ─────────────────────────────────────────────────────────────────────
        # Step 4 — EDA
        # ─────────────────────────────────────────────────────────────────────

        try:

            EDAAgent().execute(
                state
            )

        except Exception as e:

            state.log(
                f"EDA warning: {e}"
            )


        _update_progress(
            4,
            "Engineering features...",
        )


        # ─────────────────────────────────────────────────────────────────────
        # Step 5 — Feature Engineering
        # ─────────────────────────────────────────────────────────────────────

        FeatureEngineeringAgent().execute(
            state
        )


        _update_progress(
            5,
            "Training models...",
        )


        # ─────────────────────────────────────────────────────────────────────
        # Step 6 — Model Training
        # ─────────────────────────────────────────────────────────────────────

        state, candidates = MLModelAgent().execute(
            state
        )


        _update_progress(
            6,
            "Evaluating models...",
        )


        # ─────────────────────────────────────────────────────────────────────
        # Step 7 — Evaluation
        # ─────────────────────────────────────────────────────────────────────

        EvaluationAgent().evaluate(
            state,
            candidates,
        )


        _update_progress(
            7,
            "Critic audit & self-reflection...",
        )


        # ─────────────────────────────────────────────────────────────────────
        # Step 8 — Critic + Reflection
        # ─────────────────────────────────────────────────────────────────────

        orch = SelfReflectionOrchestrator(
            max_reflection_cycles=max_cycles
        )

        state, candidates = orch.run(
            state,
            candidates,
        )


        _update_progress(
            8,
            "Explaining model...",
        )


        # ─────────────────────────────────────────────────────────────────────
        # Step 9 — XAI
        # ─────────────────────────────────────────────────────────────────────

        try:

            XAIAgent().explain(
                state,
                candidates=candidates,
                save_to_research=True,
            )

        except Exception as e:

            state.log(
                f"XAI skipped: {e}"
            )


        _update_progress(
            9,
            "Generating final report...",
        )


        # ─────────────────────────────────────────────────────────────────────
        # Step 10 — Final Report
        # ─────────────────────────────────────────────────────────────────────

        FinalReportAgent().execute(
            state
        )


        _update_progress(
            10,
            "Done!",
        )


        progress_bar.progress(
            100,
            text="✅ Pipeline complete!",
        )


        st.session_state["state"] = state

        st.session_state["candidates"] = candidates


    # ─────────────────────────────────────────────────────────────────────────
    # Display Results
    # ─────────────────────────────────────────────────────────────────────────

    if "state" in st.session_state:

        state = st.session_state["state"]

        candidates = st.session_state.get(
            "candidates",
            [],
        )


        tabs = st.tabs(
            [
                "📊 Overview",
                "🏆 Leaderboard",
                "🩺 Critic & Reflection",
                "🔍 Explainability",
                "📋 Final Report",
            ]
        )


        # ═════════════════════════════════════════════════════════════════════
        # TAB 1 — OVERVIEW
        # ═════════════════════════════════════════════════════════════════════

        with tabs[0]:

            st.subheader(
                "Pipeline Overview"
            )


            col1, col2, col3, col4 = st.columns(
                4
            )


            with col1:

                st.metric(
                    "Champion Model",
                    state.champion_model_name or "—",
                )


            with col2:

                if (
                    state.evaluation_report
                    and state.evaluation_report.generalization_gap
                    is not None
                ):

                    st.metric(
                        "Generalization Gap",
                        f"{state.evaluation_report.generalization_gap:.4f}",
                    )

                else:

                    st.metric(
                        "Generalization Gap",
                        "—",
                    )


            with col3:

                st.metric(
                    "Reflection Cycles",
                    len(state.reflection_log),
                )


            with col4:

                last_fb = (
                    state.critic_history[-1]
                    if state.critic_history
                    else None
                )

                if last_fb:

                    st.metric(
                        "Critic Health",
                        f"{last_fb.health_score:.1f}/100",
                    )

                else:

                    st.metric(
                        "Critic Health",
                        "—",
                    )


            # Dataset profile

            if state.profile:

                st.markdown(
                    "### Dataset Profile"
                )

                p = state.profile

                st.json(
                    {
                        "rows": p.row_count,
                        "columns": p.col_count,
                        "target": p.target_column,
                        "problem_type": p.problem_type,
                        "class_imbalance_ratio": (
                            p.class_imbalance_ratio
                        ),
                        "potential_leakage_columns": (
                            p.potential_leakage_columns
                        ),
                    }
                )


        # ═════════════════════════════════════════════════════════════════════
        # TAB 2 — LEADERBOARD
        # ═════════════════════════════════════════════════════════════════════

        with tabs[1]:

            st.subheader(
                "Model Tournament Leaderboard"
            )

            rpt = state.evaluation_report

            if rpt and rpt.leaderboard:

                if (
                    state.profile
                    and "classification"
                    in state.profile.problem_type
                ):

                    primary = "f1_macro"

                else:

                    primary = "r2"


                rows = []


                for cand in rpt.leaderboard:

                    m = cand.metrics.get(
                        primary
                    )


                    rows.append(
                        {
                            "Model": cand.model_name,
                            "Champion": (
                                "🏆"
                                if cand.is_champion
                                else ""
                            ),
                            "Val Score": (
                                round(
                                    m.val_score,
                                    4,
                                )
                                if m
                                and m.val_score is not None
                                else None
                            ),
                            "Test Score": (
                                round(
                                    m.test_score,
                                    4,
                                )
                                if m
                                and m.test_score is not None
                                else None
                            ),
                            "CV Mean": (
                                round(
                                    m.cv_mean,
                                    4,
                                )
                                if m
                                and m.cv_mean is not None
                                else None
                            ),
                            "CV Std": (
                                round(
                                    m.cv_std,
                                    4,
                                )
                                if m
                                and m.cv_std is not None
                                else None
                            ),
                        }
                    )


                lb_df = pd.DataFrame(
                    rows
                )


                if not lb_df.empty:

                    lb_df = (
                        lb_df.sort_values(
                            "Val Score",
                            ascending=False,
                        )
                        .reset_index(drop=True)
                    )


                st.dataframe(
                    lb_df,
                    use_container_width=True,
                )

            else:

                st.info(
                    "No evaluation results yet."
                )


        # ═════════════════════════════════════════════════════════════════════
        # TAB 3 — CRITIC & REFLECTION
        # ═════════════════════════════════════════════════════════════════════

        with tabs[2]:

            st.subheader(
                "Critic Audit & Self-Reflection"
            )


            if state.critic_history:

                for fb in state.critic_history:

                    verdict_text = (
                        "✅ APPROVED"
                        if fb.verdict == "APPROVED"
                        else "❌ REJECTED"
                    )


                    with st.expander(
                        f"Iteration {fb.iteration} — "
                        f"{verdict_text} "
                        f"(Health: {fb.health_score:.1f}/100)",
                        expanded=(
                            fb.iteration
                            == len(
                                state.critic_history
                            )
                        ),
                    ):

                        audit_rows = []


                        for a in fb.audit_results:

                            audit_rows.append(
                                {
                                    "Vector": f"V{a.vector_id}",
                                    "Check": a.check_name,
                                    "Status": a.status,
                                    "Observation": (
                                        a.observation[:120]
                                    ),
                                }
                            )


                        if audit_rows:

                            st.dataframe(
                                pd.DataFrame(
                                    audit_rows
                                ),
                                use_container_width=True,
                            )


                        if fb.specific_directives:

                            st.info(
                                f"**Directives:** "
                                f"{fb.specific_directives[:300]}"
                            )

            else:

                st.info(
                    "Critic not executed."
                )


            if state.reflection_log:

                st.subheader(
                    "Self-Reflection Log"
                )


                for log in state.reflection_log:

                    icon = (
                        "✅"
                        if log.verdict == "APPROVED"
                        else "❌"
                    )


                    st.markdown(
                        f"**Cycle {log.cycle_number}** "
                        f"{icon} {log.verdict} "
                        f"(Health: {log.health_score:.1f}) "
                        f"| Action: "
                        f"{log.corrective_action or 'None'} "
                        f"| Re-run: "
                        f"{log.agent_rerun or 'None'}"
                    )


        # ═════════════════════════════════════════════════════════════════════
        # TAB 4 — EXPLAINABILITY
        # ═════════════════════════════════════════════════════════════════════

        with tabs[3]:

            st.subheader(
                "Model Explainability (XAI)"
            )


            xai = state.insight_artifact


            if xai:

                col_a, col_b = st.columns(
                    2
                )


                with col_a:

                    st.markdown(
                        f"**Explained Model:** "
                        f"`{xai.champion_model_name}`"
                    )

                    st.markdown(
                        f"**Method:** "
                        f"`{xai.explanation_method}`"
                    )


                with col_b:

                    if xai.vulnerability_warnings:

                        for w in xai.vulnerability_warnings:

                            st.warning(w)


                if xai.global_feature_importance:

                    imp_df = pd.DataFrame(
                        list(
                            xai.global_feature_importance.items()
                        )[:15],
                        columns=[
                            "Feature",
                            "Importance",
                        ],
                    ).sort_values(
                        "Importance",
                        ascending=False,
                    )


                    st.dataframe(
                        imp_df,
                        use_container_width=True,
                    )


                if xai.feature_importance_plot_path:

                    feat_img = Path(
                        xai.feature_importance_plot_path
                    )


                    if feat_img.exists():

                        st.image(
                            str(feat_img),
                            caption="Global Feature Importance",
                            use_container_width=True,
                        )


                if xai.shap_summary_plot_path:

                    shap_img = Path(
                        xai.shap_summary_plot_path
                    )


                    if shap_img.exists():

                        st.image(
                            str(shap_img),
                            caption="SHAP Summary Plot",
                            use_container_width=True,
                        )


                if xai.local_case_explanations:

                    st.subheader(
                        "Local Case Explanations"
                    )


                    for case in (
                        xai.local_case_explanations[:3]
                    ):

                        with st.expander(
                            f"Sample #{case.get('sample_index', '?')} "
                            f"— Predicted: "
                            f"{case.get('predicted_label')}"
                        ):

                            st.write(
                                f"**Actual:** "
                                f"{case.get('actual_label')} | "
                                f"**Predicted:** "
                                f"{case.get('predicted_label')} | "
                                f"**Probability:** "
                                f"{case.get('predicted_probability')}"
                            )


                            pos = case.get(
                                "top_positive_features",
                                [],
                            )


                            neg = case.get(
                                "top_negative_features",
                                [],
                            )


                            if pos:

                                st.markdown(
                                    "**Top Positive Drivers:**"
                                )


                                for f in pos:

                                    st.markdown(
                                        f"➕ `{f['feature']}`: "
                                        f"+{f['contribution']:.4f}"
                                    )


                            if neg:

                                st.markdown(
                                    "**Top Negative Drivers:**"
                                )


                                for f in neg:

                                    st.markdown(
                                        f"➖ `{f['feature']}`: "
                                        f"{f['contribution']:.4f}"
                                    )

            else:

                st.info(
                    "Explainability not generated yet."
                )


        # ═════════════════════════════════════════════════════════════════════
        # TAB 5 — FINAL REPORT
        # ═════════════════════════════════════════════════════════════════════

        with tabs[4]:

            st.subheader(
                "Final Report"
            )


            if state.final_report_path:

                report_path = Path(
                    state.final_report_path
                )


                if report_path.exists():

                    report_text = report_path.read_text(
                        encoding="utf-8"
                    )


                    st.download_button(
                        label="📥 Download Markdown Report",
                        data=report_text,
                        file_name=(
                            f"final_report_"
                            f"{state.session_id}.md"
                        ),
                        mime="text/markdown",
                        use_container_width=True,
                    )


                    with st.expander(
                        "Preview Report",
                        expanded=True,
                    ):

                        st.markdown(
                            report_text
                        )

                else:

                    st.warning(
                        "Final report path was recorded, "
                        "but the report file does not exist."
                    )

            else:

                st.info(
                    "Report not yet generated."
                )


# ─────────────────────────────────────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────────────────────────────────────

else:

    st.markdown(
        """
        ### How to use

        1. **Upload** your CSV dataset using the sidebar
        2. **Select the Target Column**
        3. **Describe** your ML objective
        4. Click **▶ Run Full Pipeline**
        5. Watch the 10-step agentic pipeline execute

        | Step | Agent | Description |
        |---|---|---|
        | 1 | DatasetIngestor | Load and validate CSV |
        | 2 | DataAnalystAgent | Profile dataset, detect issues |
        | 3 | DataCleaningAgent | Clean data, create Train/Val/Test splits |
        | 4 | EDAAgent | Exploratory analysis and charts |
        | 5 | FeatureEngineeringAgent | Encode, transform, select features |
        | 6 | MLModelAgent | Train multiple model candidates |
        | 7 | EvaluationAgent | Evaluate and rank models |
        | 8 | CriticAgent + SelfReflectionOrchestrator | Audit, detect flaws, correct and retrain |
        | 9 | XAIAgent | Generate SHAP / coefficient explanations |
        | 10 | FinalReportAgent | Produce Markdown research report |

        ### Demo datasets

        Use the sidebar buttons to load the built-in demo datasets.
        """
    )