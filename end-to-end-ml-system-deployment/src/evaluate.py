from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_pipeline import TARGET_COLUMN, clean_telco_dataframe, load_dataset, prepare_features_target
from src.infer import load_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simple monitoring report and drift artifacts.")
    parser.add_argument("--train-data-path", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--recent-batch-path", default="data/recent/recent_batch.csv")
    parser.add_argument("--model-path", default="artifacts/model_registry/final_model.joblib")
    parser.add_argument("--output-dir", default="artifacts/monitoring")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected_clean = pd.to_numeric(expected, errors="coerce").dropna()
    actual_clean = pd.to_numeric(actual, errors="coerce").dropna()

    if expected_clean.empty or actual_clean.empty:
        return np.nan

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    bucket_edges = np.unique(np.quantile(expected_clean, quantiles))
    if len(bucket_edges) < 3:
        return np.nan

    exp_counts, _ = np.histogram(expected_clean, bins=bucket_edges)
    act_counts, _ = np.histogram(actual_clean, bins=bucket_edges)

    epsilon = 1e-6
    exp_pct = (exp_counts + epsilon) / (exp_counts.sum() + epsilon * len(exp_counts))
    act_pct = (act_counts + epsilon) / (act_counts.sum() + epsilon * len(act_counts))

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def categorical_shift_score(expected: pd.Series, actual: pd.Series) -> float:
    exp_freq = expected.astype(str).value_counts(normalize=True)
    act_freq = actual.astype(str).value_counts(normalize=True)
    categories = sorted(set(exp_freq.index).union(set(act_freq.index)))

    exp_vals = np.array([exp_freq.get(cat, 0.0) for cat in categories])
    act_vals = np.array([act_freq.get(cat, 0.0) for cat in categories])

    return float(0.5 * np.abs(exp_vals - act_vals).sum())


def drift_level(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score > 0.20:
        return "high"
    if score > 0.10:
        return "medium"
    return "low"


def load_recent_batch(path: Path, train_features: pd.DataFrame, seed: int) -> pd.DataFrame:
    if path.exists():
        frame = pd.read_csv(path)
        frame = clean_telco_dataframe(frame)
        features, _ = prepare_features_target(frame)
        return features

    synthetic = train_features.sample(min(len(train_features), 200), random_state=seed).copy()
    if "MonthlyCharges" in synthetic.columns:
        synthetic["MonthlyCharges"] = synthetic["MonthlyCharges"].astype(float) * 1.10

    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(path, index=False)
    return synthetic


def build_drift_table(train_features: pd.DataFrame, recent_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    severity_rank = {"high": 3, "medium": 2, "low": 1, "unknown": 0}

    common_features = [col for col in train_features.columns if col in recent_features.columns]
    for column in common_features:
        if pd.api.types.is_numeric_dtype(train_features[column]):
            score = population_stability_index(train_features[column], recent_features[column])
            feature_type = "numeric"
        else:
            score = categorical_shift_score(train_features[column], recent_features[column])
            feature_type = "categorical"

        level = drift_level(score)
        rows.append(
            {
                "feature_name": column,
                "feature_type": feature_type,
                "drift_score": score,
                "drift_level": level,
                "_severity_rank": severity_rank[level],
            }
        )

    table = pd.DataFrame(rows).sort_values(by=["_severity_rank", "drift_score"], ascending=[False, False])
    return table.drop(columns=["_severity_rank"])


def plot_feature_distributions(
    train_features: pd.DataFrame,
    recent_features: pd.DataFrame,
    output_path: Path,
) -> None:
    numeric_cols = [col for col in train_features.columns if pd.api.types.is_numeric_dtype(train_features[col])]
    selected = numeric_cols[:4]

    if not selected:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for idx, feature in enumerate(selected):
        ax = axes[idx]
        sns.kdeplot(train_features[feature].dropna(), label="train", ax=ax, color="#1f77b4", fill=True, alpha=0.2)
        sns.kdeplot(recent_features[feature].dropna(), label="recent", ax=ax, color="#d62728", fill=True, alpha=0.2)
        ax.set_title(feature)
        ax.legend()

    for idx in range(len(selected), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Train vs Recent Feature Distribution Comparison", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(args.train_data_path)
    train_features, _ = prepare_features_target(train_df)

    recent_path = Path(args.recent_batch_path)
    recent_features = load_recent_batch(recent_path, train_features, seed=args.seed)

    runtime = load_runtime(args.model_path)
    scored = runtime.predict_records(recent_features.to_dict(orient="records"))
    scored_df = pd.DataFrame(scored)
    scored_df.to_csv(output_dir / "recent_batch_scoring.csv", index=False)

    drift_df = build_drift_table(train_features, recent_features)
    drift_df.to_csv(output_dir / "drift_indicators.csv", index=False)

    dist_plot_path = output_dir / "feature_distribution_comparison.png"
    plot_feature_distributions(train_features, recent_features, dist_plot_path)

    high_drift_features = int((drift_df["drift_level"] == "high").sum()) if not drift_df.empty else 0
    medium_drift_features = int((drift_df["drift_level"] == "medium").sum()) if not drift_df.empty else 0
    total_features = max(len(drift_df), 1)
    high_drift_ratio = high_drift_features / total_features

    avg_risk = float(scored_df["churn_probability"].mean()) if not scored_df.empty else np.nan
    high_risk_rate = float((scored_df["predicted_label"] == 1).mean()) if not scored_df.empty else np.nan

    retrain_recommended = (
        high_drift_ratio >= 0.25
        or (pd.notna(high_risk_rate) and high_risk_rate >= 0.45)
    )

    report = f"""# Monitoring Report

## Batch Summary
- Train samples: {len(train_features)}
- Recent batch samples: {len(recent_features)}
- Model source: {runtime.model_source}
- Model name: {runtime.model_name}

## Scoring Snapshot
- Mean churn probability: {avg_risk:.4f}
- High-risk prediction rate: {high_risk_rate:.4f}

## Drift Indicators
- High drift features: {high_drift_features}
- Medium drift features: {medium_drift_features}
- High drift ratio: {high_drift_ratio:.2%}

Artifacts generated:
- `drift_indicators.csv`
- `recent_batch_scoring.csv`
- `feature_distribution_comparison.png`

## Retraining Trigger Policy
Retraining should be triggered when any of the following occur:
1. More than 25% of monitored features have drift score > 0.20.
2. Expected business cost per 1000 predictions increases by at least 20% vs holdout baseline (once labels arrive).
3. At least 30 days have elapsed since the last production retrain with sufficient new labeled data.

## Current Recommendation
- Retrain now: {'YES' if retrain_recommended else 'NO'}
"""

    report_path = output_dir / "batch_scoring_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Monitoring report written to: {report_path}")
    print(f"Drift indicators written to: {output_dir / 'drift_indicators.csv'}")


if __name__ == "__main__":
    main()
