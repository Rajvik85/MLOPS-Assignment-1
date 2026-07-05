"""
Generate reproducible EDA plots for the UCI Cleveland Heart Disease dataset.
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    load_and_clean_data,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eda")


def generate_eda_plots(raw_data_path: str, output_dir: str = "plots") -> list[str]:
    """
    Create the EDA plots required by the assignment and return their file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data(raw_data_path)
    sns.set_theme(style="whitegrid", palette="deep")

    generated_files: list[str] = []

    # 1. Class balance
    class_counts = df[TARGET_COLUMN].map({0: "No disease", 1: "Disease"}).value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
    ax.set_title("Heart Disease Target Class Balance")
    ax.set_xlabel("Target class")
    ax.set_ylabel("Number of patients")
    for index, value in enumerate(class_counts.values):
        ax.text(index, value + 2, str(value), ha="center", va="bottom")
    class_balance_path = output_path / "eda_class_balance.png"
    fig.savefig(class_balance_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    generated_files.append(str(class_balance_path))

    # 2. Missing value analysis
    missing_counts = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    if missing_counts.empty:
        ax.text(0.5, 0.5, "No missing values after cleaning", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.barplot(x=missing_counts.index, y=missing_counts.values, ax=ax)
        ax.set_title("Missing Values by Feature")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Missing row count")
        for index, value in enumerate(missing_counts.values):
            ax.text(index, value + 0.05, str(value), ha="center", va="bottom")
    missing_path = output_path / "eda_missing_values.png"
    fig.savefig(missing_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    generated_files.append(str(missing_path))

    # 3. Numerical feature histograms
    hist_axes = df[NUMERICAL_FEATURES].hist(bins=18, figsize=(12, 8), color="#4c72b0")
    for axis in hist_axes.ravel():
        axis.set_ylabel("Patient count")
    plt.suptitle("Numerical Feature Distributions", y=1.02)
    hist_path = output_path / "eda_numerical_histograms.png"
    plt.savefig(hist_path, bbox_inches="tight", dpi=150)
    plt.close()
    generated_files.append(str(hist_path))

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    corr_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    corr = df[corr_columns].corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    heatmap_path = output_path / "eda_correlation_heatmap.png"
    fig.savefig(heatmap_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    generated_files.append(str(heatmap_path))

    # 5. Feature relationship analysis
    plot_df = df.copy()
    plot_df["target_label"] = plot_df[TARGET_COLUMN].map(
        {0: "No disease", 1: "Disease"}
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="age",
        y="thalach",
        hue="target_label",
        style="sex",
        alpha=0.85,
        ax=ax,
    )
    ax.set_title("Age vs Maximum Heart Rate by Heart Disease Status")
    ax.set_xlabel("Age")
    ax.set_ylabel("Maximum heart rate achieved")
    relationship_path = output_path / "eda_age_thalach_relationship.png"
    fig.savefig(relationship_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    generated_files.append(str(relationship_path))

    logger.info("Generated %s EDA plots in %s", len(generated_files), output_path)
    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EDA plots for the report.")
    parser.add_argument(
        "--data",
        default=os.path.join("data", "raw", "processed.cleveland.data"),
        help="Path to the raw UCI Cleveland dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where generated plot images will be saved.",
    )
    args = parser.parse_args()

    for file_path in generate_eda_plots(args.data, args.output_dir):
        print(file_path)
