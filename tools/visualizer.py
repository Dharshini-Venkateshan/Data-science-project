"""
Automated Visualization Generator for Exploratory Data Analysis.
Generates headless, high-resolution statistical charts saved as session artifacts.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import matplotlib
matplotlib.use("Agg")  # Set headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from core.config import WORKSPACE_DIR


class Visualizer:
    """
    Produces publication-grade EDA visualizations in PNG format.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.figures_dir = (WORKSPACE_DIR / session_id / "figures").resolve()
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Apply standard clean styling
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    def plot_missing_values(self, df: pd.DataFrame) -> Optional[str]:
        """Plots missing value percentages across columns."""
        missing = (df.isnull().sum() / len(df)) * 100.0
        missing = missing[missing > 0].sort_values(ascending=False)

        if missing.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.35)))
        sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#4A90E2")
        ax.set_title("Missing Values Percentage by Column (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Missing Percentage (%)")
        ax.set_xlim(0, 100)

        out_path = self.figures_dir / "eda_missing_values.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def plot_target_distribution(
        self,
        df: pd.DataFrame,
        target_col: str,
        problem_type: str
    ) -> Optional[str]:
        """Plots the target feature distribution (Countplot for classification, Histplot for regression)."""
        if target_col not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(7, 4.5))
        target_series = df[target_col].dropna()

        if "classification" in problem_type:
            val_counts = target_series.value_counts(ascending=False)
            x_labels = val_counts.index.astype(str)
            sns.barplot(x=x_labels, y=val_counts.values, ax=ax, hue=x_labels, palette="Blues_r", legend=False)
            ax.set_title(f"Target Distribution: {target_col} (Classification)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Count")
            ax.set_xlabel("Class")
            # Annotate percentages
            total = len(target_series)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f"{height/total:.1%}", (p.get_x() + p.get_width() / 2., height),
                            ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                            textcoords="offset points")
        else:
            sns.histplot(target_series, kde=True, ax=ax, color="#2C3E50")
            ax.set_title(f"Target Distribution: {target_col} (Regression)", fontsize=12, fontweight="bold")
            ax.set_xlabel(target_col)
            ax.set_ylabel("Frequency")

        out_path = self.figures_dir / "eda_target_distribution.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def plot_correlation_heatmap(self, df: pd.DataFrame, max_features: int = 15) -> Optional[str]:
        """Plots correlation heatmap for continuous numeric features."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None

        # Filter out constant columns
        numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]
        if numeric_df.shape[1] < 2:
            return None

        # If too many features, select the ones with highest variance
        if numeric_df.shape[1] > max_features:
            variances = numeric_df.var().sort_values(ascending=False)
            top_cols = variances.head(max_features).index
            numeric_df = numeric_df[top_cols]

        corr = numeric_df.corr(method="pearson")
        fig, ax = plt.subplots(figsize=(max(7, corr.shape[1] * 0.6), max(5, corr.shape[0] * 0.5)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            vmin=-1,
            vmax=1,
            cbar_kws={"shrink": 0.8},
            ax=ax,
            annot_kws={"size": 8}
        )
        ax.set_title("Feature Correlation Matrix (Pearson)", fontsize=12, fontweight="bold")

        out_path = self.figures_dir / "eda_correlation_matrix.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def plot_feature_vs_target(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        problem_type: str
    ) -> Optional[str]:
        """Generates a relationship plot between a key feature and the target."""
        if feature_col not in df.columns or target_col not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(6, 4))
        if "classification" in problem_type and pd.api.types.is_numeric_dtype(df[feature_col]):
            t_labels = df[target_col].astype(str)
            sns.boxplot(x=t_labels, y=df[feature_col], ax=ax, hue=t_labels, palette="Set2", legend=False)
            ax.set_title(f"{feature_col} by {target_col}", fontsize=11, fontweight="bold")
        elif pd.api.types.is_numeric_dtype(df[feature_col]) and pd.api.types.is_numeric_dtype(df[target_col]):
            sns.scatterplot(x=df[feature_col], y=df[target_col], alpha=0.6, ax=ax, color="#16A085")
            ax.set_title(f"{target_col} vs {feature_col}", fontsize=11, fontweight="bold")
        else:
            plt.close(fig)
            return None

        safe_name = "".join(c for c in feature_col if c.isalnum() or c == "_")
        out_path = self.figures_dir / f"eda_relationship_{safe_name}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
