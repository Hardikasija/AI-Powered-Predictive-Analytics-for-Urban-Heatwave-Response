import os
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_time_series(df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    sample = df.sort_values("timestamp").groupby("timestamp")["temperature"].mean()
    sample.plot(ax=ax)
    ax.set_title("Average Temperature Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (C)")
    path = os.path.join(output_dir, "time_series.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_cols: list, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if not hasattr(model, "feature_importances_"):
        return ""

    importance = model.feature_importances_
    df = pd.DataFrame({"feature": feature_cols, "importance": importance}).sort_values(
        "importance", ascending=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df.head(10), x="importance", y="feature", ax=ax)
    ax.set_title(f"Feature Importance - {name}")
    path = os.path.join(output_dir, f"feature_importance_{name}.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_model_metrics(metrics: Dict[str, dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(metrics).T
    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(kind="bar", ax=ax)
    ax.set_title("Model Metrics")
    ax.set_ylabel("Score")
    path = os.path.join(output_dir, "model_metrics.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
