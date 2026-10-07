"""
ML Model Agent for Autonomous AI Data Scientist.
Selects candidate algorithms, executes k-fold cross-validation, optimizes hyperparameters via Optuna,
trains candidates, evaluates on validation data, and serializes trained model artifacts.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import optuna

# Scikit-Learn Algorithms & CV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error

# Modern Gradient Boosting
import lightgbm as lgb
import xgboost as xgb

from agents.base import BaseAgent
from core.state import DataScienceState, TrainedModelCandidate, ModelEvaluationMetric
from core.llm_client import BaseLLMClient
from core.config import WORKSPACE_DIR

# Silence Optuna info logs for cleaner console output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class MLModelAgent(BaseAgent):
    """
    Orchestrates algorithm candidate selection, cross-validation, and Optuna tuning.
    Keeps the test partition strictly untouched.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        super().__init__(name="MLModelAgent", llm_client=llm_client)

    def _get_candidate_models(self, problem_type: str, n_samples: int) -> Dict[str, Any]:
        """
        Dynamically selects model candidates tailored to the dataset scale and problem type.
        """
        candidates = {}

        if "classification" in problem_type:
            # Baseline Linear Classifier
            candidates["LogisticRegression"] = LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced"
            )
            # Ensemble Bagging
            candidates["RandomForest"] = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42, class_weight="balanced", n_jobs=-1
            )
            # Gradient Boosted Trees
            candidates["HistGradientBoosting"] = HistGradientBoostingClassifier(
                random_state=42, max_iter=100
            )
            candidates["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, random_state=42, verbose=-1, n_jobs=-1
            )
            # Add XGBoost if dataset has reasonable size
            if n_samples >= 30:
                candidates["XGBoost"] = xgb.XGBClassifier(
                    n_estimators=100, max_depth=5, random_state=42, eval_metric="logloss", n_jobs=-1
                )

        else:  # Regression
            candidates["Ridge"] = Ridge(alpha=1.0, random_state=42)
            candidates["RandomForest"] = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
            )
            candidates["HistGradientBoosting"] = HistGradientBoostingRegressor(
                random_state=42, max_iter=100
            )
            candidates["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, random_state=42, verbose=-1, n_jobs=-1
            )
            if n_samples >= 30:
                candidates["XGBoost"] = xgb.XGBRegressor(
                    n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
                )

        return candidates

    def _optimize_with_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        problem_type: str,
        n_trials: int = 5
    ) -> Dict[str, Any]:
        """
        Executes a targeted Optuna Bayesian optimization study on LightGBM.
        """
        is_classification = "classification" in problem_type
        scoring_metric = "f1_macro" if is_classification else "r2"

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1
            }

            if is_classification:
                model = lgb.LGBMClassifier(**params)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            else:
                model = lgb.LGBMRegressor(**params)
                cv = KFold(n_splits=3, shuffle=True, random_state=42)

            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_metric)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def execute(self, state: DataScienceState) -> DataScienceState:
        """
        Trains model candidates using cross-validation and validation set evaluation.
        """
        if not state.feature_artifact or not state.feature_artifact.engineered_train_path:
            raise ValueError("MLModelAgent requires engineered_train_path in state.")

        # 1. Load Engineered Train and Validation Data (TEST SET REMAINS UNTOUCHED)
        train_df = pd.read_parquet(state.feature_artifact.engineered_train_path)
        val_df = pd.read_parquet(state.feature_artifact.engineered_val_path)

        target_col = state.profile.target_column
        problem_type = state.profile.problem_type

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]

        models_dir = (WORKSPACE_DIR / state.session_id / "models").resolve()
        models_dir.mkdir(parents=True, exist_ok=True)

        # ── Read corrective directives from Critic / Orchestrator ──────────
        directives = state.critic_directives or {}
        stronger_regularization  = bool(directives.get("stronger_regularization",  False))
        enhanced_class_balance   = bool(directives.get("enhanced_class_balance",   False))
        increase_cv_folds        = bool(directives.get("increase_cv_folds",        False))
        extra_cv_splits          = int(directives.get("n_cv_splits", 7))

        # 2. Select Candidates & Set Up Cross-Validation
        candidate_models = self._get_candidate_models(problem_type, len(X_train))

        # ── Directive: stronger regularisation ────────────────────────────
        if stronger_regularization:
            is_clf = "classification" in problem_type
            if is_clf:
                candidate_models["LogisticRegression"] = LogisticRegression(
                    C=0.05, max_iter=1000, random_state=42, class_weight="balanced"
                )
                candidate_models["RandomForest"] = RandomForestClassifier(
                    n_estimators=100, max_depth=4, min_samples_leaf=10,
                    random_state=42, class_weight="balanced", n_jobs=-1
                )
                candidate_models["HistGradientBoosting"] = HistGradientBoostingClassifier(
                    max_depth=3, min_samples_leaf=15, random_state=42, max_iter=100
                )
                candidate_models["LightGBM"] = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=4, num_leaves=15, min_child_samples=20,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1
                )
            else:
                candidate_models["Ridge"] = Ridge(alpha=10.0, random_state=42)
                candidate_models["RandomForest"] = RandomForestRegressor(
                    n_estimators=100, max_depth=4, min_samples_leaf=10,
                    random_state=42, n_jobs=-1
                )
            state.log("MLModelAgent (corrective): applied stronger_regularization directive.")

        # ── Directive: enhanced class balance (SMOTE-style oversampling) ──
        if enhanced_class_balance and "classification" in problem_type:
            try:
                from imblearn.over_sampling import SMOTE  # type: ignore
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                state.log(
                    f"MLModelAgent (corrective): SMOTE oversampling applied. "
                    f"New training size = {len(X_train)}."
                )
            except ImportError:
                # imbalanced-learn not installed; fall back to stronger class_weight only
                state.log(
                    "MLModelAgent (corrective): imbalanced-learn not available; "
                    "class_weight='balanced' already applied by default models."
                )

        is_classification = "classification" in problem_type
        primary_metric_name = "f1_macro" if is_classification else "r2"

        if is_classification:
            min_class_count = int(y_train.value_counts().min())
            base_splits = min(5, max(2, min_class_count))
            # ── Directive: increase CV folds ───────────────────────────────
            if increase_cv_folds:
                n_splits = max(base_splits, extra_cv_splits)
                state.log(f"MLModelAgent (corrective): CV folds increased to {n_splits}.")
            else:
                n_splits = base_splits
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            base_splits = 5
            if increase_cv_folds:
                n_splits = max(base_splits, extra_cv_splits)
                state.log(f"MLModelAgent (corrective): CV folds increased to {n_splits}.")
            else:
                n_splits = base_splits
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 3. Optuna Hyperparameter Optimization on Top GBDT Candidate
        best_optuna_params = self._optimize_with_optuna(X_train, y_train, problem_type, n_trials=5)
        if is_classification:
            candidate_models["LightGBM_Tuned"] = lgb.LGBMClassifier(**best_optuna_params)
        else:
            candidate_models["LightGBM_Tuned"] = lgb.LGBMRegressor(**best_optuna_params)

        evaluated_candidates: List[TrainedModelCandidate] = []

        # 4. Train, Cross-Validate, and Evaluate on Validation Data
        for name, model in candidate_models.items():
            # A. Cross-Validation on Training Set
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=primary_metric_name)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            except Exception:
                cv_mean = 0.0
                cv_std = 0.0

            # B. Fit on Full Training Set
            model.fit(X_train, y_train)

            # C. Evaluate on Validation Set (NOT Test!)
            val_preds = model.predict(X_val)
            train_preds = model.predict(X_train)

            if is_classification:
                val_score = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
                train_score = float(f1_score(y_train, train_preds, average="macro", zero_division=0))
            else:
                val_score = float(r2_score(y_val, val_preds))
                train_score = float(r2_score(y_train, train_preds))

            # D. Serialize Trained Model Artifact
            model_path = models_dir / f"{name}.joblib"
            joblib.dump(model, model_path)

            metric_entry = ModelEvaluationMetric(
                metric_name=primary_metric_name,
                train_score=round(train_score, 4),
                val_score=round(val_score, 4),
                cv_mean=round(cv_mean, 4),
                cv_std=round(cv_std, 4)
            )

            candidate_record = TrainedModelCandidate(
                model_name=name,
                algorithm=type(model).__name__,
                hyperparameters={k: str(v) for k, v in model.get_params().items() if k in ["n_estimators", "max_depth", "learning_rate", "alpha", "C"]},
                metrics={primary_metric_name: metric_entry},
                artifact_path=str(model_path)
            )
            evaluated_candidates.append(candidate_record)

        # Store candidate records temporarily in state logs / plan for Evaluation Agent
        state.log(f"MLModelAgent: Trained and cross-validated {len(evaluated_candidates)} candidate models.")
        state.status = "MODELING"

        # Pass candidates to Evaluation Engine via state attribute
        # We can store raw candidates in state.models_evaluated or prepare EvaluationReport
        return state, evaluated_candidates
