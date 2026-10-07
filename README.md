# 🤖 Autonomous AI Data Scientist

A research-grade, multi-agent autonomous data science system that takes a raw CSV file and a natural-language ML objective, then executes a complete, self-correcting, explainable data science pipeline.

---

## Architecture Overview

```
USER
 │
 ▼
UPLOAD CSV  +  NATURAL LANGUAGE GOAL
 │
 ▼
┌─────────────────────────────────────────────────┐
│           AUTONOMOUS PIPELINE                   │
│                                                 │
│  DataAnalystAgent  ──► DataCleaningAgent        │
│        │                     │                  │
│        ▼                     ▼                  │
│     EDAAgent    ──► FeatureEngineeringAgent     │
│                              │                  │
│                              ▼                  │
│                    MLModelAgent (Tournament)    │
│                              │                  │
│                              ▼                  │
│                     EvaluationAgent             │
│                              │                  │
│                              ▼                  │
│              ┌─── CriticAgent (8 Vectors) ───┐  │
│              │     APPROVED / REJECTED        │  │
│              │            │                  │  │
│              │            ▼ (if REJECTED)    │  │
│              │   SelfReflectionOrchestrator  │  │
│              │   (Corrective reruns loop)    │  │
│              └───────────────────────────────┘  │
│                              │                  │
│                              ▼                  │
│                  XAIAgent (SHAP / Coefficients) │
│                              │                  │
│                              ▼                  │
│                   FinalReportAgent              │
└─────────────────────────────────────────────────┘
 │
 ▼
RESULTS + ARTIFACTS + FINAL REPORT
```

---

## Phases

| Phase | Scope | Key Agents |
|---|---|---|
| 1 | Data Ingestion & Profiling | `DatasetIngestor`, `DataAnalystAgent` |
| 2 | Data Cleaning & EDA | `DataCleaningAgent`, `EDAAgent` |
| 3 | Feature Engineering & Model Training | `FeatureEngineeringAgent`, `MLModelAgent`, `EvaluationAgent` |
| 4 | Critic Audit & Self-Reflection | `CriticAgent`, `SelfReflectionOrchestrator` |
| 5+ | Explainability, Ablation, Reporting | `XAIAgent`, `AblationStudy`, `FinalReportAgent` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API key (optional — LLM features only)

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

> All ML pipeline steps work without an API key. LLM features (analyst summaries, critic reasoning) are gracefully degraded when no key is present.

### 3. Run the web UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload a CSV, set your goal, and click **▶ Run Full Pipeline**.

### 4. Run the CLI demo

```bash
# Full pipeline on the built-in credit risk dataset
python demo_final.py

# Custom dataset and goal
python demo_final.py --csv data/my_data.csv --goal "Predict churn. Maximize F1 macro."

# All options
python demo_final.py --help
```

### 5. Run ablation / research experiments

```bash
python demo_phase5.py
```

Runs the 3-arm ablation study (Baseline / +Critic / Full System) and generates research artifacts under `research/`.

---

## Repository Structure

```
autonomous_datascientist/
├── agents/                     # All agent implementations
│   ├── analyst.py              # DataAnalystAgent — profiling & goal extraction
│   ├── cleaner.py              # DataCleaningAgent — cleaning + train/val/test split
│   ├── eda.py                  # EDAAgent — correlation, distributions, charts
│   ├── feature_engineer.py     # FeatureEngineeringAgent — encoding, transformations
│   ├── model_trainer.py        # MLModelAgent — model tournament
│   ├── evaluator.py            # EvaluationAgent — comprehensive metrics
│   ├── critic.py               # CriticAgent — 8-vector adversarial audit
│   ├── xai.py                  # XAIAgent — SHAP / coefficient explanations
│   └── reporter.py             # FinalReportAgent — Markdown report generation
├── core/
│   ├── state.py                # Pydantic state models (DataScienceState, etc.)
│   ├── orchestrator.py         # SelfReflectionOrchestrator — closed-loop correction
│   ├── config.py               # Config & paths
│   ├── llm_client.py           # LLM client (Gemini / mock fallback)
│   └── sandbox.py              # Safe code execution sandbox
├── tools/
│   ├── ingestion.py            # DatasetIngestor
│   ├── profiler.py             # Column profiling utilities
│   └── visualizer.py          # Chart rendering utilities
├── research/
│   ├── ablation.py             # AblationStudy — 3-arm reproducible experiments
│   ├── experiments/            # experiment_log.json
│   ├── results/                # ablation_results.csv
│   ├── figures/                # feature_importance.png, shap_summary.png, etc.
│   └── reports/                # Research report outputs
├── tests/                      # Pytest test suite (52+ tests)
├── workspace/                  # Session artifacts (models, processed data, figures)
├── app.py                      # Streamlit web UI
├── demo_final.py               # CLI end-to-end demo
├── demo_phase5.py              # Ablation & XAI research demo
├── demo_phase4.py              # Critic & self-reflection demo
├── requirements.txt
└── .env.example
```

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Phase-specific
pytest tests/test_ingestion.py -v
pytest tests/test_profiler_and_cleaning.py -v
pytest tests/test_modeling_and_evaluation.py -v
pytest tests/test_critic_and_orchestration.py -v
pytest tests/test_xai_and_ablation.py -v
```

Expected: **52+ tests passing, 0 failures**.

---

## Critic Agent — 8 Diagnostic Vectors

| Vector | Check |
|---|---|
| V1 | Target Leakage Detection |
| V2 | Overfitting / Generalization Gap |
| V3 | Class Imbalance & Calibration |
| V4 | High CV Variance |
| V5 | Unexpectedly High Score (potential leakage) |
| V6 | Residual Heteroscedasticity (regression) |
| V7 | Dominant Single Feature |
| V8 | Dataset Size Adequacy |

---

## XAI / Explainability

| Model Family | Primary Method | Fallback |
|---|---|---|
| Tree-based (RF, XGB, LGBM, HGB) | TreeSHAP | `feature_importances_` |
| Linear (LR, Ridge, Lasso) | LinearSHAP | Normalized coefficients |

**Outputs:**
- Global feature importance bar chart (`feature_importance.png`)
- SHAP summary plot (`shap_summary.png`)
- Local case explanations (3 representative validation samples)
- Vulnerability warnings (dominant proxy feature detection)

> **Data integrity**: the test set is never accessed during XAI analysis. Explainers are fitted only on train/val background data.

---

## Ablation Study

Compares 3 architectural configurations on identical data splits:

| Arm | Critic | Self-Reflection | Purpose |
|---|---|---|---|
| A | ✗ | ✗ | Standard ML Baseline |
| B | ✅ | ✗ | Audit-only — identifies but does not fix |
| C | ✅ | ✅ | Full proposed system |

Results saved to `research/results/ablation_results.csv` and `research/experiments/experiment_log.json`.

> **Research integrity**: results are reported exactly as measured. No test information leaks into model selection.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key (optional) |
| `GEMINI_REASONING_MODEL` | `gemini-2.5-pro` | Model for deep reasoning |
| `GEMINI_FAST_MODEL` | `gemini-2.5-flash` | Model for fast generation |
| `SANDBOX_DEFAULT_TIMEOUT` | `60` | Sandbox execution timeout (seconds) |
| `SANDBOX_MAX_MEMORY_MB` | `4096` | Sandbox memory limit |

---

## Requirements

- Python 3.10+
- Key packages: `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `pandas`, `numpy`, `matplotlib`, `streamlit`, `pydantic>=2`, `joblib`, `pyarrow`
- Optional: `google-generativeai` (for LLM features)

Install all:
```bash
pip install -r requirements.txt
```

---

## Research Notes

- **Single-run limitation**: Statistical significance tests require ≥30 repeated runs with different seeds. Single-run results are indicative only.
- **No test set contamination**: all model selection, hyperparameter tuning, critic decisions, and XAI analyses use only train and validation splits.
- **Reproducibility**: fixed random seed (`42`) across all pipeline components. Ablation arms share identical underlying data splits.
