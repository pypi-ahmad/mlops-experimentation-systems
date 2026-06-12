from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
import mlflow
import numpy as np
import pandas as pd
from flaml import AutoML
from lazypredict.Supervised import LazyClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_pipeline import (
    TARGET_COLUMN,
    build_preprocessor,
    load_dataset,
    split_dataset,
)

warnings.filterwarnings("ignore")


def _suppress_resource_tracker_childprocesserror() -> None:
    try:
        import multiprocessing.resource_tracker as resource_tracker
    except Exception:
        return

    original_del = getattr(resource_tracker.ResourceTracker, "__del__", None)
    if original_del is None:
        return

    def safe_del(self: Any) -> None:
        try:
            original_del(self)
        except ChildProcessError:
            # Non-fatal third-party teardown issue; suppress noisy destructor traceback.
            return

    resource_tracker.ResourceTracker.__del__ = safe_del


_suppress_resource_tracker_childprocesserror()

PROJECT_NAME = "end-to-end-ml-system-deployment"
TASK_TYPE = "binary_classification"
PRIMARY_METRIC_NAME = "expected_business_cost_per_1000"
SECONDARY_METRIC_NAME = "pr_auc"
TERTIARY_METRIC_NAME = "roc_auc"

REQUIRED_COLUMNS = [
    "project_name",
    "task_type",
    "library_source",
    "model_name",
    "cv_metric_mean",
    "cv_metric_std",
    "holdout_primary_metric",
    "holdout_secondary_metric",
    "holdout_tertiary_metric",
    "calibration_metric",
    "train_time_sec",
    "infer_latency_ms",
    "model_size_mb",
    "interpretability_note",
    "rank_score",
    "final_rank",
]

EXPORT_COLUMNS = REQUIRED_COLUMNS + ["p95_latency_ms", "retrain_time_sec", "_candidate_id"]

LAZY_TO_MANUAL_FAMILY = {
    "LogisticRegression": "logistic_regression",
    "RidgeClassifier": "logistic_regression",
    "LinearDiscriminantAnalysis": "logistic_regression",
    "Perceptron": "logistic_regression",
    "RandomForestClassifier": "random_forest",
    "ExtraTreesClassifier": "random_forest",
    "DecisionTreeClassifier": "random_forest",
    "BaggingClassifier": "random_forest",
    "XGBClassifier": "xgboost",
    "LGBMClassifier": "xgboost",
    "GradientBoostingClassifier": "xgboost",
    "AdaBoostClassifier": "xgboost",
}

MANUAL_FAMILY_NOTES = {
    "logistic_regression": "Linear baseline; high interpretability and stable calibration.",
    "random_forest": "Non-linear ensemble; robust interactions, moderate interpretability.",
    "xgboost": "Boosted trees; strong tabular performance, lower explainability without SHAP tooling.",
}

_FLAML_FP_COST = 65.0
_FLAML_FN_COST = 320.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and rank churn models with 4 serious labs: "
            "LazyPredict discovery, manual engineering, FLAML optimization, and PyCaret experiments."
        )
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to the Telco churn CSV dataset.",
    )
    parser.add_argument("--experiment-name", default="e2e-ml-system")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manual-top-n", type=int, default=3)
    parser.add_argument("--lazy-top-k", type=int, default=12)
    parser.add_argument("--flaml-time-budget", type=int, default=90)
    parser.add_argument("--fp-cost", type=float, default=65.0)
    parser.add_argument("--fn-cost", type=float, default=320.0)
    parser.add_argument("--max-p95-latency-ms", type=float, default=40.0)
    parser.add_argument("--min-secondary-metric", type=float, default=0.45)
    parser.add_argument("--max-calibration-metric", type=float, default=0.30)
    parser.add_argument("--run-stability-check", action="store_true")
    parser.add_argument("--stability-seeds", default="11,42,87")
    return parser.parse_args()


def parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    return seeds or [42]


def maybe_mlflow_run(enabled: bool, run_name: str) -> contextlib.AbstractContextManager:
    if enabled:
        return mlflow.start_run(run_name=run_name, nested=True)
    return contextlib.nullcontext()


def safe_log_artifact(path: Path, enabled: bool) -> None:
    if enabled and path.exists():
        mlflow.log_artifact(str(path))


def safe_log_metrics(metrics: dict[str, float], enabled: bool) -> None:
    if not enabled:
        return
    safe_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None and np.isfinite(value)
    }
    if safe_metrics:
        mlflow.log_metrics(safe_metrics)


def safe_log_params(params: dict[str, Any], enabled: bool) -> None:
    if enabled:
        mlflow.log_params(params)


def expected_business_cost_per_1000(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fp_cost: float,
    fn_cost: float,
) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = (fp * fp_cost) + (fn * fn_cost)
    return float((total_cost / max(len(y_true), 1)) * 1000.0)


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return np.nan


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan


def safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return np.nan


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fp_cost: float,
    fn_cost: float,
) -> tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.05, 0.95, 181)
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        cost = expected_business_cost_per_1000(y_true, y_pred, fp_cost, fn_cost)
        precision = float(np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1))
        recall = float(np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1))
        rows.append(
            {
                "threshold": float(threshold),
                "expected_cost_per_1000": cost,
                "precision": precision,
                "recall": recall,
            }
        )

    tradeoff_df = pd.DataFrame(rows).sort_values(
        by=["expected_cost_per_1000", "recall"],
        ascending=[True, False],
    )
    best_threshold = float(tradeoff_df.iloc[0]["threshold"])
    return best_threshold, tradeoff_df


def predict_probabilities(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        probs = np.asarray(probs)
        if probs.ndim == 2:
            return probs[:, 1].astype(float)
        return probs.astype(float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        scores = np.clip(scores, -20, 20)
        return 1.0 / (1.0 + np.exp(-scores))

    preds = np.asarray(model.predict(X), dtype=float)
    return np.clip(preds, 0.0, 1.0)


def measure_latency_ms(
    predictor: Callable[[pd.DataFrame], np.ndarray],
    sample_frame: pd.DataFrame,
    runs: int = 120,
) -> tuple[float, float]:
    if sample_frame.empty:
        return np.nan, np.nan

    latencies_ms: list[float] = []
    for idx in range(runs):
        row = sample_frame.iloc[[idx % len(sample_frame)]].copy()
        start = time.perf_counter()
        _ = predictor(row)
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed)

    return float(np.mean(latencies_ms)), float(np.percentile(latencies_ms, 95))


def estimate_model_size_mb(obj: Any) -> float:
    try:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=True) as tmp:
            joblib.dump(obj, tmp.name)
            size_bytes = Path(tmp.name).stat().st_size
        return float(size_bytes / (1024.0 * 1024.0))
    except Exception:
        return np.nan


def build_record(
    *,
    library_source: str,
    model_name: str,
    cv_metric_mean: float,
    cv_metric_std: float,
    holdout_primary_metric: float,
    holdout_secondary_metric: float,
    holdout_tertiary_metric: float,
    calibration_metric: float,
    train_time_sec: float,
    infer_latency_ms: float,
    p95_latency_ms: float,
    model_size_mb: float,
    retrain_time_sec: float,
    interpretability_note: str,
    candidate_id: str,
) -> dict[str, Any]:
    return {
        "project_name": PROJECT_NAME,
        "task_type": TASK_TYPE,
        "library_source": library_source,
        "model_name": model_name,
        "cv_metric_mean": float(cv_metric_mean) if pd.notna(cv_metric_mean) else np.nan,
        "cv_metric_std": float(cv_metric_std) if pd.notna(cv_metric_std) else np.nan,
        "holdout_primary_metric": float(holdout_primary_metric) if pd.notna(holdout_primary_metric) else np.nan,
        "holdout_secondary_metric": float(holdout_secondary_metric)
        if pd.notna(holdout_secondary_metric)
        else np.nan,
        "holdout_tertiary_metric": float(holdout_tertiary_metric)
        if pd.notna(holdout_tertiary_metric)
        else np.nan,
        "calibration_metric": float(calibration_metric) if pd.notna(calibration_metric) else np.nan,
        "train_time_sec": float(train_time_sec) if pd.notna(train_time_sec) else np.nan,
        "infer_latency_ms": float(infer_latency_ms) if pd.notna(infer_latency_ms) else np.nan,
        "p95_latency_ms": float(p95_latency_ms) if pd.notna(p95_latency_ms) else np.nan,
        "model_size_mb": float(model_size_mb) if pd.notna(model_size_mb) else np.nan,
        "retrain_time_sec": float(retrain_time_sec) if pd.notna(retrain_time_sec) else np.nan,
        "interpretability_note": interpretability_note,
        "_candidate_id": candidate_id,
    }


def minmax_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(values)

    if not finite_mask.any():
        return pd.Series(np.zeros_like(values, dtype=float), index=series.index)

    finite_values = values[finite_mask]
    min_val = finite_values.min()
    max_val = finite_values.max()

    if np.isclose(min_val, max_val):
        normalized = np.ones_like(values, dtype=float) * 0.5
    else:
        normalized = (values - min_val) / (max_val - min_val)

    if not higher_is_better:
        normalized = 1.0 - normalized

    normalized[~finite_mask] = 0.0
    return pd.Series(normalized, index=series.index)


def enrich_with_ranking(leaderboard: pd.DataFrame) -> pd.DataFrame:
    scored = leaderboard.copy()

    primary_score = minmax_score(scored["holdout_primary_metric"], higher_is_better=False)
    secondary_score = minmax_score(scored["holdout_secondary_metric"], higher_is_better=True)
    tertiary_score = minmax_score(scored["holdout_tertiary_metric"], higher_is_better=True)
    calibration_score = minmax_score(scored["calibration_metric"], higher_is_better=False)

    latency_score = minmax_score(scored["infer_latency_ms"], higher_is_better=False)
    size_score = minmax_score(scored["model_size_mb"], higher_is_better=False)
    retrain_score = minmax_score(scored["retrain_time_sec"], higher_is_better=False)

    quality_component = (
        (0.55 * primary_score)
        + (0.20 * secondary_score)
        + (0.15 * tertiary_score)
        + (0.10 * calibration_score)
    )
    ops_component = (0.45 * latency_score) + (0.35 * size_score) + (0.20 * retrain_score)

    scored["rank_score"] = ((0.75 * quality_component) + (0.25 * ops_component)) * 100.0
    scored["final_rank"] = scored["rank_score"].rank(ascending=False, method="first").astype(int)
    return scored.sort_values("final_rank").reset_index(drop=True)


def select_winner(
    leaderboard: pd.DataFrame,
    deployable_ids: set[str],
    max_p95_latency_ms: float,
    min_secondary_metric: float,
    max_calibration_metric: float,
) -> tuple[pd.Series, pd.DataFrame]:
    table = leaderboard.copy()
    table["_guardrail_pass"] = (
        table["p95_latency_ms"].fillna(np.inf).le(max_p95_latency_ms)
        & table["holdout_secondary_metric"].fillna(-np.inf).ge(min_secondary_metric)
        & table["calibration_metric"].fillna(np.inf).le(max_calibration_metric)
    )
    table["_deployable"] = table["_candidate_id"].isin(deployable_ids)

    eligible = table[(table["_guardrail_pass"]) & (table["_deployable"])].sort_values(
        "rank_score", ascending=False
    )
    if not eligible.empty:
        return eligible.iloc[0], table

    fallback = table[table["_deployable"]].sort_values("rank_score", ascending=False)
    if fallback.empty:
        raise RuntimeError("No deployable candidate artifacts were produced.")
    return fallback.iloc[0], table


def model_from_family(family: str, seed: int) -> Any:
    if family == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if family == "xgboost":
        return XGBClassifier(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )

    raise ValueError(f"Unsupported manual family: {family}")


def evaluate_fitted_model(
    *,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fp_cost: float,
    fn_cost: float,
    cv_seed: int,
) -> dict[str, Any]:
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=cv_seed),
        scoring="average_precision",
        n_jobs=1,
    )

    valid_probs = predict_probabilities(model, X_valid)
    best_threshold, tradeoff_df = optimize_threshold(y_valid.to_numpy(), valid_probs, fp_cost, fn_cost)

    test_probs = predict_probabilities(model, X_test)
    test_preds = (test_probs >= best_threshold).astype(int)

    holdout_primary = expected_business_cost_per_1000(y_test.to_numpy(), test_preds, fp_cost, fn_cost)
    holdout_secondary = safe_average_precision(y_test.to_numpy(), test_probs)
    holdout_tertiary = safe_roc_auc(y_test.to_numpy(), test_probs)
    calibration_metric = safe_brier(y_test.to_numpy(), test_probs)

    infer_latency, p95_latency = measure_latency_ms(
        predictor=lambda frame: predict_probabilities(model, frame),
        sample_frame=X_test,
    )

    tn, fp, fn, tp = confusion_matrix(y_test.to_numpy(), test_preds, labels=[0, 1]).ravel()
    error_summary = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(best_threshold),
        "expected_cost_per_1000": float(holdout_primary),
    }

    return {
        "fitted_model": model,
        "tradeoff_df": tradeoff_df,
        "cv_metric_mean": float(np.mean(cv_scores)),
        "cv_metric_std": float(np.std(cv_scores)),
        "holdout_primary_metric": holdout_primary,
        "holdout_secondary_metric": holdout_secondary,
        "holdout_tertiary_metric": holdout_tertiary,
        "calibration_metric": calibration_metric,
        "train_time_sec": train_time,
        "infer_latency_ms": infer_latency,
        "p95_latency_ms": p95_latency,
        "best_threshold": float(best_threshold),
        "error_summary": error_summary,
    }


def run_baseline_track(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fp_cost: float,
    fn_cost: float,
    artifacts_dir: Path,
    seed: int,
    log_mlflow: bool,
    persist_artifacts: bool,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Path] = {}

    with maybe_mlflow_run(log_mlflow, "baseline_dummy"):
        baseline_pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", DummyClassifier(strategy="most_frequent", random_state=seed)),
            ]
        )

        eval_result = evaluate_fitted_model(
            model=baseline_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            cv_seed=seed,
        )

        candidate_id = "baseline::dummy_most_frequent"
        model_bundle = {
            "model": eval_result["fitted_model"],
            "threshold": eval_result["best_threshold"],
            "feature_columns": X_train.columns.tolist(),
            "source": "baseline",
            "model_name": "dummy_most_frequent",
        }

        model_size_mb = estimate_model_size_mb(model_bundle)
        model_path = artifacts_dir / "models" / "baseline_dummy_most_frequent.joblib"
        threshold_path = artifacts_dir / "reports" / "baseline_dummy_threshold_tradeoff.csv"

        if persist_artifacts:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            threshold_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_bundle, model_path)
            eval_result["tradeoff_df"].to_csv(threshold_path, index=False)
            artifacts[candidate_id] = model_path
            safe_log_artifact(model_path, log_mlflow)
            safe_log_artifact(threshold_path, log_mlflow)

        safe_log_params(
            {
                "library_source": "baseline",
                "model_name": "dummy_most_frequent",
                "best_threshold": eval_result["best_threshold"],
            },
            log_mlflow,
        )
        safe_log_metrics(
            {
                "cost_per_1000": eval_result["holdout_primary_metric"],
                "pr_auc": eval_result["holdout_secondary_metric"],
                "roc_auc": eval_result["holdout_tertiary_metric"],
            },
            log_mlflow,
        )

        records.append(
            build_record(
                library_source="baseline",
                model_name="dummy_most_frequent",
                cv_metric_mean=eval_result["cv_metric_mean"],
                cv_metric_std=eval_result["cv_metric_std"],
                holdout_primary_metric=eval_result["holdout_primary_metric"],
                holdout_secondary_metric=eval_result["holdout_secondary_metric"],
                holdout_tertiary_metric=eval_result["holdout_tertiary_metric"],
                calibration_metric=eval_result["calibration_metric"],
                train_time_sec=eval_result["train_time_sec"],
                infer_latency_ms=eval_result["infer_latency_ms"],
                p95_latency_ms=eval_result["p95_latency_ms"],
                model_size_mb=model_size_mb,
                retrain_time_sec=eval_result["train_time_sec"],
                interpretability_note="Reference naive baseline for business-cost comparison.",
                candidate_id=candidate_id,
            )
        )

    return records, artifacts


def _detect_lazy_column(columns: Iterable[str], preferred: Sequence[str]) -> str | None:
    available = list(columns)
    for candidate in preferred:
        if candidate in available:
            return candidate
    return None


def run_lazypredict_discovery_lab(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    fp_cost: float,
    fn_cost: float,
    top_k: int,
    manual_top_n: int,
    artifacts_dir: Path,
    log_mlflow: bool,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []

    with maybe_mlflow_run(log_mlflow, "lazy_discovery_lab"):
        preprocessor = build_preprocessor(X_train)
        X_train_encoded = preprocessor.fit_transform(X_train)
        X_valid_encoded = preprocessor.transform(X_valid)

        if hasattr(X_train_encoded, "toarray"):
            X_train_encoded = X_train_encoded.toarray()
            X_valid_encoded = X_valid_encoded.toarray()

        lazy = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
        start = time.perf_counter()
        models_df, predictions_df = lazy.fit(X_train_encoded, X_valid_encoded, y_train, y_valid)
        total_time = time.perf_counter() - start

        ranked = models_df.reset_index().rename(columns={"index": "model_name"})

        roc_col = _detect_lazy_column(ranked.columns, ["ROC AUC", "ROC_AUC", "ROC Auc"])
        ba_col = _detect_lazy_column(ranked.columns, ["Balanced Accuracy", "Balanced_Accuracy", "Accuracy"])
        f1_col = _detect_lazy_column(ranked.columns, ["F1 Score", "F1_Score", "F1"])
        time_col = _detect_lazy_column(ranked.columns, ["Time Taken", "Time_Taken"])

        if roc_col is None:
            ranked["_roc_auc"] = np.nan
        else:
            ranked["_roc_auc"] = pd.to_numeric(ranked[roc_col], errors="coerce")

        if ba_col is None:
            ranked["_balanced_accuracy"] = np.nan
        else:
            ranked["_balanced_accuracy"] = pd.to_numeric(ranked[ba_col], errors="coerce")

        if f1_col is None:
            ranked["_f1"] = np.nan
        else:
            ranked["_f1"] = pd.to_numeric(ranked[f1_col], errors="coerce")

        if time_col is None:
            ranked["_time_taken"] = total_time / max(len(ranked), 1)
        else:
            ranked["_time_taken"] = pd.to_numeric(ranked[time_col], errors="coerce")

        ranked["manual_family"] = ranked["model_name"].astype(str).map(LAZY_TO_MANUAL_FAMILY)
        ranked["eligible_for_manual"] = (
            ranked["manual_family"].notna()
            & ranked["_roc_auc"].fillna(-np.inf).ge(0.60)
            & ranked["_balanced_accuracy"].fillna(-np.inf).ge(0.55)
        )

        ranked = ranked.sort_values(by=["_roc_auc", "_balanced_accuracy"], ascending=False).reset_index(drop=True)

        selected_rows: list[dict[str, Any]] = []
        seen_families: set[str] = set()

        for _, row in ranked[ranked["eligible_for_manual"]].iterrows():
            family = str(row["manual_family"])
            if family in seen_families:
                continue
            selected_rows.append(
                {
                    "lazy_rank": len(selected_rows) + 1,
                    "lazy_model_name": row["model_name"],
                    "manual_family": family,
                    "eligibility_reason": "Meets quality floor and is implementable in manual pipeline.",
                    "lazy_roc_auc": row["_roc_auc"],
                    "lazy_balanced_accuracy": row["_balanced_accuracy"],
                }
            )
            seen_families.add(family)
            if len(selected_rows) >= manual_top_n:
                break

        if len(selected_rows) < manual_top_n:
            fallback = ranked[ranked["manual_family"].notna()]
            for _, row in fallback.iterrows():
                family = str(row["manual_family"])
                if family in seen_families:
                    continue
                selected_rows.append(
                    {
                        "lazy_rank": len(selected_rows) + 1,
                        "lazy_model_name": row["model_name"],
                        "manual_family": family,
                        "eligibility_reason": "Fallback: implementable family added to ensure 3 manual tracks.",
                        "lazy_roc_auc": row["_roc_auc"],
                        "lazy_balanced_accuracy": row["_balanced_accuracy"],
                    }
                )
                seen_families.add(family)
                if len(selected_rows) >= manual_top_n:
                    break

        for family in ["logistic_regression", "random_forest", "xgboost"]:
            if len(selected_rows) >= manual_top_n:
                break
            if family in seen_families:
                continue
            selected_rows.append(
                {
                    "lazy_rank": len(selected_rows) + 1,
                    "lazy_model_name": "fallback_placeholder",
                    "manual_family": family,
                    "eligibility_reason": "Fallback placeholder: no suitable LazyPredict mapping available.",
                    "lazy_roc_auc": np.nan,
                    "lazy_balanced_accuracy": np.nan,
                }
            )
            seen_families.add(family)

        top3_eligible = pd.DataFrame(selected_rows).head(manual_top_n)

        benchmark = ranked.head(top_k).copy()
        for _, row in benchmark.iterrows():
            model_name = str(row["model_name"])
            y_pred = None
            if model_name in predictions_df.columns:
                y_pred = predictions_df[model_name].astype(int).to_numpy()

            holdout_primary = (
                expected_business_cost_per_1000(y_valid.to_numpy(), y_pred, fp_cost, fn_cost)
                if y_pred is not None
                else np.nan
            )

            records.append(
                build_record(
                    library_source="lazypredict",
                    model_name=model_name,
                    cv_metric_mean=row["_balanced_accuracy"],
                    cv_metric_std=np.nan,
                    holdout_primary_metric=holdout_primary,
                    holdout_secondary_metric=row["_roc_auc"],
                    holdout_tertiary_metric=row["_f1"],
                    calibration_metric=np.nan,
                    train_time_sec=row["_time_taken"],
                    infer_latency_ms=np.nan,
                    p95_latency_ms=np.nan,
                    model_size_mb=np.nan,
                    retrain_time_sec=row["_time_taken"],
                    interpretability_note=(
                        "Discovery benchmark only; used for model-family screening before manual implementation."
                    ),
                    candidate_id=f"lazypredict::{model_name}",
                )
            )

        ranked_path = artifacts_dir / "reports" / "lazypredict_ranked_benchmark.csv"
        pred_path = artifacts_dir / "reports" / "lazypredict_validation_predictions.csv"
        top3_path = artifacts_dir / "reports" / "lazypredict_top3_eligible.csv"
        ranked_path.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(ranked_path, index=False)
        predictions_df.to_csv(pred_path, index=False)
        top3_eligible.to_csv(top3_path, index=False)

        safe_log_params({"lazy_models_scanned": len(ranked), "manual_top_n": manual_top_n}, log_mlflow)
        safe_log_metrics({"lazy_sweep_time_sec": total_time}, log_mlflow)
        safe_log_artifact(ranked_path, log_mlflow)
        safe_log_artifact(top3_path, log_mlflow)

    return records, ranked, top3_eligible


def run_manual_engineering_lab(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_families: Sequence[str],
    lazy_top3: pd.DataFrame,
    fp_cost: float,
    fn_cost: float,
    artifacts_dir: Path,
    seed: int,
    log_mlflow: bool,
    persist_artifacts: bool,
) -> tuple[list[dict[str, Any]], dict[str, Path], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Path] = {}
    diagnostics_rows: list[dict[str, Any]] = []

    selected_lookup = {}
    if not lazy_top3.empty:
        selected_lookup = (
            lazy_top3.set_index("manual_family")["lazy_model_name"].astype(str).to_dict()
        )

    for family in selected_families:
        with maybe_mlflow_run(log_mlflow, f"manual_engineering_{family}"):
            base_model = model_from_family(family, seed)
            base_pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(X_train)),
                    ("model", clone(base_model)),
                ]
            )

            if family in {"random_forest", "xgboost"}:
                model = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=3)
            else:
                model = base_pipeline

            eval_result = evaluate_fitted_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
                cv_seed=seed,
            )

            candidate_id = f"manual::{family}"
            model_bundle = {
                "model": eval_result["fitted_model"],
                "threshold": eval_result["best_threshold"],
                "feature_columns": X_train.columns.tolist(),
                "source": "manual",
                "model_name": family,
                "selected_from_lazypredict": selected_lookup.get(family),
            }

            model_size_mb = estimate_model_size_mb(model_bundle)

            model_path = artifacts_dir / "models" / f"manual_{family}.joblib"
            tradeoff_path = artifacts_dir / "reports" / f"manual_{family}_threshold_tradeoff.csv"
            errors_path = artifacts_dir / "reports" / f"manual_{family}_error_analysis.csv"

            if persist_artifacts:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                tradeoff_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model_bundle, model_path)
                eval_result["tradeoff_df"].to_csv(tradeoff_path, index=False)
                pd.DataFrame([eval_result["error_summary"]]).to_csv(errors_path, index=False)
                artifacts[candidate_id] = model_path
                safe_log_artifact(model_path, log_mlflow)
                safe_log_artifact(tradeoff_path, log_mlflow)
                safe_log_artifact(errors_path, log_mlflow)

            interpretability_note = (
                f"Manual family from LazyPredict candidate: {selected_lookup.get(family, 'fallback')}. "
                f"{MANUAL_FAMILY_NOTES.get(family, '')}"
            )

            safe_log_params(
                {
                    "library_source": "manual",
                    "model_name": family,
                    "selected_from_lazypredict": selected_lookup.get(family, "fallback"),
                    "best_threshold": eval_result["best_threshold"],
                },
                log_mlflow,
            )
            safe_log_metrics(
                {
                    "cost_per_1000": eval_result["holdout_primary_metric"],
                    "pr_auc": eval_result["holdout_secondary_metric"],
                    "roc_auc": eval_result["holdout_tertiary_metric"],
                    "brier": eval_result["calibration_metric"],
                    "train_time_sec": eval_result["train_time_sec"],
                    "p95_latency_ms": eval_result["p95_latency_ms"],
                },
                log_mlflow,
            )

            records.append(
                build_record(
                    library_source="manual",
                    model_name=family,
                    cv_metric_mean=eval_result["cv_metric_mean"],
                    cv_metric_std=eval_result["cv_metric_std"],
                    holdout_primary_metric=eval_result["holdout_primary_metric"],
                    holdout_secondary_metric=eval_result["holdout_secondary_metric"],
                    holdout_tertiary_metric=eval_result["holdout_tertiary_metric"],
                    calibration_metric=eval_result["calibration_metric"],
                    train_time_sec=eval_result["train_time_sec"],
                    infer_latency_ms=eval_result["infer_latency_ms"],
                    p95_latency_ms=eval_result["p95_latency_ms"],
                    model_size_mb=model_size_mb,
                    retrain_time_sec=eval_result["train_time_sec"],
                    interpretability_note=interpretability_note,
                    candidate_id=candidate_id,
                )
            )

            diagnostics_rows.append(
                {
                    "candidate_id": candidate_id,
                    "library_source": "manual",
                    "model_name": family,
                    "selected_from_lazypredict": selected_lookup.get(family, "fallback"),
                    **eval_result["error_summary"],
                    "holdout_pr_auc": eval_result["holdout_secondary_metric"],
                    "holdout_roc_auc": eval_result["holdout_tertiary_metric"],
                    "calibration_metric": eval_result["calibration_metric"],
                }
            )

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    if persist_artifacts and not diagnostics_df.empty:
        diagnostics_path = artifacts_dir / "reports" / "manual_engineering_diagnostics.csv"
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_csv(diagnostics_path, index=False)

    return records, artifacts, diagnostics_df


def flaml_business_cost_metric(
    X_val: Any,
    y_val: Any,
    estimator: Any,
    labels: Any,
    X_train: Any,
    y_train: Any,
    weight_val: Any = None,
    weight_train: Any = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[float, dict[str, float]]:
    y_true = np.asarray(y_val).astype(int)
    probs = predict_probabilities(estimator, X_val)
    preds = (probs >= 0.5).astype(int)
    cost = expected_business_cost_per_1000(y_true, preds, _FLAML_FP_COST, _FLAML_FN_COST)
    return cost, {
        "pr_auc": safe_average_precision(y_true, probs),
        "roc_auc": safe_roc_auc(y_true, probs),
    }


def run_flaml_optimization_lab(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fp_cost: float,
    fn_cost: float,
    time_budget: int,
    artifacts_dir: Path,
    seed: int,
    log_mlflow: bool,
    persist_artifacts: bool,
) -> tuple[list[dict[str, Any]], dict[str, Path], pd.DataFrame, dict[str, Any]]:
    global _FLAML_FP_COST, _FLAML_FN_COST

    records: list[dict[str, Any]] = []
    artifacts: dict[str, Path] = {}

    candidate_id = "flaml::automl_best"

    with maybe_mlflow_run(log_mlflow, "flaml_optimization_lab"):
        automl = AutoML()
        _FLAML_FP_COST = float(fp_cost)
        _FLAML_FN_COST = float(fn_cost)
        custom_metric = flaml_business_cost_metric

        settings = {
            "time_budget": time_budget,
            "task": "classification",
            "metric": custom_metric,
            "eval_method": "cv",
            "n_splits": 3,
            "seed": seed,
            "log_file_name": str(artifacts_dir / "logs" / "flaml.log"),
            "verbose": 0,
            "n_jobs": 1,
            "estimator_list": ["xgboost", "rf", "extra_tree", "xgb_limitdepth", "lgbm"],
        }

        start = time.perf_counter()
        try:
            automl.fit(X_train=X_train, y_train=y_train, **settings)
            metric_mode = "business_cost"
        except Exception:
            fallback_settings = settings.copy()
            fallback_settings["metric"] = "roc_auc"
            fallback_settings.pop("estimator_list", None)
            automl.fit(X_train=X_train, y_train=y_train, **fallback_settings)
            metric_mode = "roc_auc_fallback"
        train_time = time.perf_counter() - start

        valid_probs = predict_probabilities(automl, X_valid)
        best_threshold, tradeoff_df = optimize_threshold(y_valid.to_numpy(), valid_probs, fp_cost, fn_cost)

        test_probs = predict_probabilities(automl, X_test)
        test_preds = (test_probs >= best_threshold).astype(int)

        holdout_primary = expected_business_cost_per_1000(y_test.to_numpy(), test_preds, fp_cost, fn_cost)
        holdout_secondary = safe_average_precision(y_test.to_numpy(), test_probs)
        holdout_tertiary = safe_roc_auc(y_test.to_numpy(), test_probs)
        calibration_metric = safe_brier(y_test.to_numpy(), test_probs)

        infer_latency, p95_latency = measure_latency_ms(
            predictor=lambda frame: predict_probabilities(automl, frame),
            sample_frame=X_test,
        )

        search_rows: list[dict[str, Any]] = []
        loss_by_estimator = getattr(automl, "best_loss_per_estimator", {}) or {}
        config_by_estimator = getattr(automl, "best_config_per_estimator", {}) or {}

        for estimator_name in sorted(set(loss_by_estimator).union(set(config_by_estimator))):
            search_rows.append(
                {
                    "estimator": estimator_name,
                    "best_loss": loss_by_estimator.get(estimator_name),
                    "best_config": json.dumps(config_by_estimator.get(estimator_name, {})),
                }
            )

        search_summary = pd.DataFrame(search_rows)
        best_config_payload = {
            "metric_mode": metric_mode,
            "best_estimator": str(getattr(automl, "best_estimator", "unknown")),
            "best_config": getattr(automl, "best_config", {}),
            "best_loss": getattr(automl, "best_loss", None),
            "time_budget": time_budget,
            "estimator_list": settings.get("estimator_list"),
        }

        model_bundle = {
            "model": automl,
            "threshold": best_threshold,
            "feature_columns": X_train.columns.tolist(),
            "source": "flaml",
            "model_name": "automl_best",
            "best_estimator": str(getattr(automl, "best_estimator", "unknown")),
            "best_config": getattr(automl, "best_config", {}),
        }

        model_size_mb = estimate_model_size_mb(model_bundle)

        model_path = artifacts_dir / "models" / "flaml_automl_best.joblib"
        tradeoff_path = artifacts_dir / "reports" / "flaml_threshold_tradeoff.csv"
        search_path = artifacts_dir / "reports" / "flaml_search_summary.csv"
        config_path = artifacts_dir / "reports" / "flaml_best_config.json"
        persist_note_path = artifacts_dir / "reports" / "flaml_persistence_warning.txt"

        if persist_artifacts:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            search_path.parent.mkdir(parents=True, exist_ok=True)
            flaml_persist_ok = True
            try:
                joblib.dump(model_bundle, model_path)
                if persist_note_path.exists():
                    persist_note_path.unlink()
            except Exception as exc:
                flaml_persist_ok = False
                if model_path.exists():
                    model_path.unlink()
                persist_note_path.write_text(
                    (
                        "FLAML model artifact could not be serialized due to non-picklable state.\\n"
                        f"Error: {exc}\\n"
                        "Track metrics and search summary remain valid; deployability guardrail will exclude this candidate."
                    ),
                    encoding="utf-8",
                )
                safe_log_artifact(persist_note_path, log_mlflow)
            tradeoff_df.to_csv(tradeoff_path, index=False)
            search_summary.to_csv(search_path, index=False)
            config_path.write_text(json.dumps(best_config_payload, indent=2), encoding="utf-8")
            if flaml_persist_ok:
                artifacts[candidate_id] = model_path
                safe_log_artifact(model_path, log_mlflow)
            safe_log_artifact(tradeoff_path, log_mlflow)
            safe_log_artifact(search_path, log_mlflow)
            safe_log_artifact(config_path, log_mlflow)

        safe_log_params(
            {
                "library_source": "flaml",
                "metric_mode": metric_mode,
                "best_estimator": str(getattr(automl, "best_estimator", "unknown")),
                "time_budget_sec": time_budget,
                "best_threshold": best_threshold,
            },
            log_mlflow,
        )
        safe_log_metrics(
            {
                "cost_per_1000": holdout_primary,
                "pr_auc": holdout_secondary,
                "roc_auc": holdout_tertiary,
                "brier": calibration_metric,
                "train_time_sec": train_time,
                "p95_latency_ms": p95_latency,
            },
            log_mlflow,
        )

        cv_metric_mean = np.nan
        cv_metric_std = np.nan
        if getattr(automl, "best_loss", None) is not None and np.isfinite(automl.best_loss):
            cv_metric_mean = float(-automl.best_loss)

        records.append(
            build_record(
                library_source="flaml",
                model_name="automl_best",
                cv_metric_mean=cv_metric_mean,
                cv_metric_std=cv_metric_std,
                holdout_primary_metric=holdout_primary,
                holdout_secondary_metric=holdout_secondary,
                holdout_tertiary_metric=holdout_tertiary,
                calibration_metric=calibration_metric,
                train_time_sec=train_time,
                infer_latency_ms=infer_latency,
                p95_latency_ms=p95_latency,
                model_size_mb=model_size_mb,
                retrain_time_sec=train_time,
                interpretability_note=(
                    "FLAML optimization lab: searched multiple learners with explicit time budget "
                    "and project business-cost objective."
                ),
                candidate_id=candidate_id,
            )
        )

    return records, artifacts, search_summary, best_config_payload


def extract_pycaret_score_column(pred_df: pd.DataFrame) -> tuple[str | None, str | None]:
    score_candidates = ["prediction_score", "Score", "prediction_score_1"]
    label_candidates = ["prediction_label", "Label", "prediction_label_1"]

    score_col = next((col for col in score_candidates if col in pred_df.columns), None)
    label_col = next((col for col in label_candidates if col in pred_df.columns), None)
    return score_col, label_col


def pycaret_cv_stats(table: pd.DataFrame, metric_name: str = "AUC") -> tuple[float, float]:
    if table is None or table.empty or metric_name not in table.columns:
        return np.nan, np.nan

    series = pd.to_numeric(table[metric_name], errors="coerce")
    series = series.dropna()
    if series.empty:
        return np.nan, np.nan

    return float(series.mean()), float(series.std())


def run_pycaret_experiment_lab(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fp_cost: float,
    fn_cost: float,
    artifacts_dir: Path,
    seed: int,
    log_mlflow: bool,
    persist_artifacts: bool,
) -> tuple[list[dict[str, Any]], dict[str, Path], dict[str, pd.DataFrame]]:
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {}

    candidate_id = "pycaret::experiment_finalized"

    try:
        from pycaret.classification import (
            calibrate_model,
            compare_models,
            finalize_model,
            predict_model,
            pull,
            save_model,
            setup,
            tune_model,
        )
    except Exception as exc:
        records.append(
            build_record(
                library_source="pycaret",
                model_name="unavailable",
                cv_metric_mean=np.nan,
                cv_metric_std=np.nan,
                holdout_primary_metric=np.inf,
                holdout_secondary_metric=np.nan,
                holdout_tertiary_metric=np.nan,
                calibration_metric=np.nan,
                train_time_sec=0.0,
                infer_latency_ms=np.nan,
                p95_latency_ms=np.nan,
                model_size_mb=np.nan,
                retrain_time_sec=0.0,
                interpretability_note=f"PyCaret import failed: {exc}",
                candidate_id=candidate_id,
            )
        )
        return records, artifacts, tables

    with maybe_mlflow_run(log_mlflow, "pycaret_experiment_lab"):
        train_df = X_train.copy()
        train_df[TARGET_COLUMN] = y_train.values

        setup_kwargs = {
            "data": train_df,
            "target": TARGET_COLUMN,
            "session_id": seed,
            "fold": 3,
            "html": False,
            "verbose": False,
            "n_jobs": 1,
        }

        start = time.perf_counter()
        try:
            setup(**setup_kwargs)
        except TypeError:
            setup_kwargs.pop("n_jobs", None)
            setup(**setup_kwargs)

        try:
            compared = compare_models(sort="AUC", n_select=3, turbo=True, errors="ignore")
        except TypeError:
            compared = compare_models(sort="AUC", n_select=3, turbo=True)

        if not isinstance(compared, list):
            compared = [compared]

        compare_table = pull()
        tables["compare"] = compare_table.copy()

        leading_model = compared[0]

        try:
            tuned_model = tune_model(leading_model, optimize="AUC", choose_better=True)
            tune_table = pull()
        except Exception:
            tuned_model = leading_model
            tune_table = pd.DataFrame()
        tables["tune"] = tune_table.copy()

        try:
            calibrated_model = calibrate_model(tuned_model, method="sigmoid")
            calibration_table = pull()
        except Exception:
            calibrated_model = tuned_model
            calibration_table = pd.DataFrame()
        tables["calibration"] = calibration_table.copy()

        final_model = finalize_model(calibrated_model)
        train_time = time.perf_counter() - start

        valid_pred_df = predict_model(final_model, data=X_valid.copy())
        test_pred_df = predict_model(final_model, data=X_test.copy())

        score_col, label_col = extract_pycaret_score_column(valid_pred_df)

        if score_col is not None:
            valid_probs = valid_pred_df[score_col].astype(float).to_numpy()
        else:
            valid_probs = predict_probabilities(final_model, X_valid)

        best_threshold, tradeoff_df = optimize_threshold(y_valid.to_numpy(), valid_probs, fp_cost, fn_cost)

        test_score_col, _ = extract_pycaret_score_column(test_pred_df)
        if test_score_col is not None:
            test_probs = test_pred_df[test_score_col].astype(float).to_numpy()
        else:
            test_probs = predict_probabilities(final_model, X_test)

        test_preds = (test_probs >= best_threshold).astype(int)
        holdout_primary = expected_business_cost_per_1000(y_test.to_numpy(), test_preds, fp_cost, fn_cost)
        holdout_secondary = safe_average_precision(y_test.to_numpy(), test_probs)
        holdout_tertiary = safe_roc_auc(y_test.to_numpy(), test_probs)
        calibration_metric = safe_brier(y_test.to_numpy(), test_probs)

        infer_latency, p95_latency = measure_latency_ms(
            predictor=lambda frame: predict_probabilities(final_model, frame),
            sample_frame=X_test,
        )

        model_bundle = {
            "model": final_model,
            "threshold": best_threshold,
            "feature_columns": X_train.columns.tolist(),
            "source": "pycaret",
            "model_name": "experiment_finalized",
        }

        model_size_mb = estimate_model_size_mb(model_bundle)

        model_path = artifacts_dir / "models" / "pycaret_experiment_finalized.joblib"
        pycaret_native_path = artifacts_dir / "models" / "pycaret_experiment_finalized"
        tradeoff_path = artifacts_dir / "reports" / "pycaret_threshold_tradeoff.csv"
        compare_path = artifacts_dir / "reports" / "pycaret_compare_table.csv"
        tune_path = artifacts_dir / "reports" / "pycaret_tune_table.csv"
        calibration_path = artifacts_dir / "reports" / "pycaret_calibration_table.csv"

        if persist_artifacts:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            compare_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_bundle, model_path)
            save_model(final_model, str(pycaret_native_path))
            tradeoff_df.to_csv(tradeoff_path, index=False)
            if not compare_table.empty:
                compare_table.to_csv(compare_path, index=False)
            if not tune_table.empty:
                tune_table.to_csv(tune_path, index=False)
            if not calibration_table.empty:
                calibration_table.to_csv(calibration_path, index=False)
            artifacts[candidate_id] = model_path

            safe_log_artifact(model_path, log_mlflow)
            safe_log_artifact(tradeoff_path, log_mlflow)
            safe_log_artifact(compare_path, log_mlflow)
            safe_log_artifact(tune_path, log_mlflow)
            safe_log_artifact(calibration_path, log_mlflow)

        cv_metric_mean, cv_metric_std = pycaret_cv_stats(tune_table, metric_name="AUC")
        if pd.isna(cv_metric_mean):
            cv_metric_mean, cv_metric_std = pycaret_cv_stats(compare_table, metric_name="AUC")

        safe_log_params(
            {
                "library_source": "pycaret",
                "model_name": "experiment_finalized",
                "best_threshold": best_threshold,
            },
            log_mlflow,
        )
        safe_log_metrics(
            {
                "cost_per_1000": holdout_primary,
                "pr_auc": holdout_secondary,
                "roc_auc": holdout_tertiary,
                "brier": calibration_metric,
                "train_time_sec": train_time,
                "p95_latency_ms": p95_latency,
            },
            log_mlflow,
        )

        records.append(
            build_record(
                library_source="pycaret",
                model_name="experiment_finalized",
                cv_metric_mean=cv_metric_mean,
                cv_metric_std=cv_metric_std,
                holdout_primary_metric=holdout_primary,
                holdout_secondary_metric=holdout_secondary,
                holdout_tertiary_metric=holdout_tertiary,
                calibration_metric=calibration_metric,
                train_time_sec=train_time,
                infer_latency_ms=infer_latency,
                p95_latency_ms=p95_latency,
                model_size_mb=model_size_mb,
                retrain_time_sec=train_time,
                interpretability_note=(
                    "PyCaret experiment lab using setup -> compare_models -> tune_model -> "
                    "calibrate_model -> finalize_model."
                ),
                candidate_id=candidate_id,
            )
        )

    return records, artifacts, tables


def run_seed_stability_check(
    *,
    top_candidates: pd.DataFrame,
    seeds: Sequence[int],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fp_cost: float,
    fn_cost: float,
    flaml_time_budget: int,
    artifacts_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, candidate in top_candidates.iterrows():
        source = str(candidate["library_source"])
        model_name = str(candidate["model_name"])

        for seed in seeds:
            result_record: dict[str, Any] | None = None
            status = "ok"

            try:
                if source == "manual":
                    manual_records, _, _ = run_manual_engineering_lab(
                        X_train=X_train,
                        y_train=y_train,
                        X_valid=X_valid,
                        y_valid=y_valid,
                        X_test=X_test,
                        y_test=y_test,
                        selected_families=[model_name],
                        lazy_top3=pd.DataFrame(),
                        fp_cost=fp_cost,
                        fn_cost=fn_cost,
                        artifacts_dir=artifacts_dir,
                        seed=seed,
                        log_mlflow=False,
                        persist_artifacts=False,
                    )
                    result_record = manual_records[0] if manual_records else None

                elif source == "flaml":
                    flaml_records, _, _, _ = run_flaml_optimization_lab(
                        X_train=X_train,
                        y_train=y_train,
                        X_valid=X_valid,
                        y_valid=y_valid,
                        X_test=X_test,
                        y_test=y_test,
                        fp_cost=fp_cost,
                        fn_cost=fn_cost,
                        time_budget=max(25, min(45, flaml_time_budget)),
                        artifacts_dir=artifacts_dir,
                        seed=seed,
                        log_mlflow=False,
                        persist_artifacts=False,
                    )
                    result_record = flaml_records[0] if flaml_records else None

                elif source == "pycaret":
                    pycaret_records, _, _ = run_pycaret_experiment_lab(
                        X_train=X_train,
                        y_train=y_train,
                        X_valid=X_valid,
                        y_valid=y_valid,
                        X_test=X_test,
                        y_test=y_test,
                        fp_cost=fp_cost,
                        fn_cost=fn_cost,
                        artifacts_dir=artifacts_dir,
                        seed=seed,
                        log_mlflow=False,
                        persist_artifacts=False,
                    )
                    result_record = pycaret_records[0] if pycaret_records else None

                else:
                    status = "skipped_not_retrainable"

            except Exception as exc:
                status = f"failed: {exc}"

            rows.append(
                {
                    "candidate_id": candidate.get("_candidate_id"),
                    "library_source": source,
                    "model_name": model_name,
                    "seed": seed,
                    "status": status,
                    "holdout_primary_metric": result_record.get("holdout_primary_metric")
                    if result_record
                    else np.nan,
                    "holdout_secondary_metric": result_record.get("holdout_secondary_metric")
                    if result_record
                    else np.nan,
                    "holdout_tertiary_metric": result_record.get("holdout_tertiary_metric")
                    if result_record
                    else np.nan,
                    "calibration_metric": result_record.get("calibration_metric") if result_record else np.nan,
                    "infer_latency_ms": result_record.get("infer_latency_ms") if result_record else np.nan,
                }
            )

    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _prepare_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "model_registry").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)


def _persist_split_snapshots(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> None:
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_snapshot = X_train.copy()
    train_snapshot[TARGET_COLUMN] = y_train.values
    train_snapshot.to_csv(processed_dir / "train_split.csv", index=False)

    holdout_snapshot = X_test.copy()
    holdout_snapshot[TARGET_COLUMN] = y_test.values
    holdout_snapshot.to_csv(processed_dir / "holdout_split.csv", index=False)

    recent_dir = Path("data/recent")
    recent_dir.mkdir(parents=True, exist_ok=True)
    recent_batch = X_test.sample(min(len(X_test), 250), random_state=seed).copy()
    if "MonthlyCharges" in recent_batch.columns:
        recent_batch["MonthlyCharges"] = recent_batch["MonthlyCharges"].astype(float) * 1.08
    recent_batch.to_csv(recent_dir / "recent_batch.csv", index=False)


def run_full_training_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    _prepare_output_dirs(output_dir)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset file not found. Expected path: "
            f"{data_path}. Use this download command first:\n"
            "uv run --with kaggle kaggle datasets download "
            "-d blastchar/telco-customer-churn -p data/raw --unzip"
        )

    df = load_dataset(data_path)
    splits = split_dataset(df, random_state=args.seed)
    _persist_split_snapshots(
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_test=splits.X_test,
        y_test=splits.y_test,
        seed=args.seed,
    )

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment_name)

    all_records: list[dict[str, Any]] = []
    artifact_map: dict[str, Path] = {}

    with mlflow.start_run(run_name="training_orchestration"):
        safe_log_params(
            {
                "project_name": PROJECT_NAME,
                "task_type": TASK_TYPE,
                "dataset_path": str(data_path),
                "fp_cost": args.fp_cost,
                "fn_cost": args.fn_cost,
                "flaml_time_budget": args.flaml_time_budget,
                "manual_top_n": args.manual_top_n,
            },
            True,
        )

        baseline_records, baseline_artifacts = run_baseline_track(
            X_train=splits.X_train,
            y_train=splits.y_train,
            X_valid=splits.X_valid,
            y_valid=splits.y_valid,
            X_test=splits.X_test,
            y_test=splits.y_test,
            fp_cost=args.fp_cost,
            fn_cost=args.fn_cost,
            artifacts_dir=output_dir,
            seed=args.seed,
            log_mlflow=True,
            persist_artifacts=True,
        )
        all_records.extend(baseline_records)
        artifact_map.update(baseline_artifacts)

        lazy_records, lazy_ranked, lazy_top3 = run_lazypredict_discovery_lab(
            X_train=splits.X_train,
            y_train=splits.y_train,
            X_valid=splits.X_valid,
            y_valid=splits.y_valid,
            fp_cost=args.fp_cost,
            fn_cost=args.fn_cost,
            top_k=args.lazy_top_k,
            manual_top_n=args.manual_top_n,
            artifacts_dir=output_dir,
            log_mlflow=True,
        )
        all_records.extend(lazy_records)

        selected_families = lazy_top3["manual_family"].dropna().astype(str).tolist()[: args.manual_top_n]

        manual_records, manual_artifacts, manual_diag = run_manual_engineering_lab(
            X_train=splits.X_train,
            y_train=splits.y_train,
            X_valid=splits.X_valid,
            y_valid=splits.y_valid,
            X_test=splits.X_test,
            y_test=splits.y_test,
            selected_families=selected_families,
            lazy_top3=lazy_top3,
            fp_cost=args.fp_cost,
            fn_cost=args.fn_cost,
            artifacts_dir=output_dir,
            seed=args.seed,
            log_mlflow=True,
            persist_artifacts=True,
        )
        all_records.extend(manual_records)
        artifact_map.update(manual_artifacts)

        flaml_records, flaml_artifacts, flaml_search_summary, flaml_config = run_flaml_optimization_lab(
            X_train=splits.X_train,
            y_train=splits.y_train,
            X_valid=splits.X_valid,
            y_valid=splits.y_valid,
            X_test=splits.X_test,
            y_test=splits.y_test,
            fp_cost=args.fp_cost,
            fn_cost=args.fn_cost,
            time_budget=args.flaml_time_budget,
            artifacts_dir=output_dir,
            seed=args.seed,
            log_mlflow=True,
            persist_artifacts=True,
        )
        all_records.extend(flaml_records)
        artifact_map.update(flaml_artifacts)

        pycaret_records, pycaret_artifacts, pycaret_tables = run_pycaret_experiment_lab(
            X_train=splits.X_train,
            y_train=splits.y_train,
            X_valid=splits.X_valid,
            y_valid=splits.y_valid,
            X_test=splits.X_test,
            y_test=splits.y_test,
            fp_cost=args.fp_cost,
            fn_cost=args.fn_cost,
            artifacts_dir=output_dir,
            seed=args.seed,
            log_mlflow=True,
            persist_artifacts=True,
        )
        all_records.extend(pycaret_records)
        artifact_map.update(pycaret_artifacts)

        leaderboard = pd.DataFrame(all_records)
        leaderboard = enrich_with_ranking(leaderboard)

        winner, scoring_table = select_winner(
            leaderboard=leaderboard,
            deployable_ids=set(artifact_map.keys()),
            max_p95_latency_ms=args.max_p95_latency_ms,
            min_secondary_metric=args.min_secondary_metric,
            max_calibration_metric=args.max_calibration_metric,
        )

        selected_artifact = artifact_map[winner["_candidate_id"]]
        final_model_path = output_dir / "model_registry" / "final_model.joblib"
        shutil.copy2(selected_artifact, final_model_path)

        leaderboard_path = output_dir / "leaderboard_e2e.csv"
        export_table = leaderboard.copy()
        for column in EXPORT_COLUMNS:
            if column not in export_table.columns:
                export_table[column] = np.nan
        export_table[EXPORT_COLUMNS].to_csv(leaderboard_path, index=False)

        lazy_top3_path = output_dir / "reports" / "lazypredict_top3_eligible.csv"
        manual_diag_path = output_dir / "reports" / "manual_engineering_diagnostics.csv"

        report_payload = {
            "project_name": PROJECT_NAME,
            "task_type": TASK_TYPE,
            "selection_rule": "Best rank_score subject to latency and reliability guardrails.",
            "metrics": {
                "primary": PRIMARY_METRIC_NAME,
                "secondary": SECONDARY_METRIC_NAME,
                "tertiary": TERTIARY_METRIC_NAME,
            },
            "lazy_to_manual_rule": "Only top-3 eligible LazyPredict families are manually implemented.",
            "selected_manual_families": selected_families,
            "winner": {
                "library_source": winner["library_source"],
                "model_name": winner["model_name"],
                "candidate_id": winner["_candidate_id"],
                "rank_score": float(winner["rank_score"]),
                "holdout_primary_metric": float(winner["holdout_primary_metric"]),
                "holdout_secondary_metric": float(winner["holdout_secondary_metric"]),
                "holdout_tertiary_metric": float(winner["holdout_tertiary_metric"])
                if pd.notna(winner["holdout_tertiary_metric"])
                else None,
                "p95_latency_ms": float(winner["p95_latency_ms"])
                if pd.notna(winner["p95_latency_ms"])
                else None,
                "calibration_metric": float(winner["calibration_metric"])
                if pd.notna(winner["calibration_metric"])
                else None,
            },
            "guardrails": {
                "max_p95_latency_ms": args.max_p95_latency_ms,
                "min_secondary_metric": args.min_secondary_metric,
                "max_calibration_metric": args.max_calibration_metric,
            },
            "cost_assumptions": {
                "false_positive_cost": args.fp_cost,
                "false_negative_cost": args.fn_cost,
            },
            "artifacts": {
                "leaderboard": str(leaderboard_path),
                "final_model": str(final_model_path),
                "lazy_top3": str(lazy_top3_path),
                "manual_diagnostics": str(manual_diag_path),
                "flaml_search_summary": str(output_dir / "reports" / "flaml_search_summary.csv"),
                "flaml_best_config": str(output_dir / "reports" / "flaml_best_config.json"),
                "pycaret_compare": str(output_dir / "reports" / "pycaret_compare_table.csv"),
                "pycaret_tune": str(output_dir / "reports" / "pycaret_tune_table.csv"),
            },
            "guardrail_table": scoring_table[
                [
                    "library_source",
                    "model_name",
                    "rank_score",
                    "_guardrail_pass",
                    "_deployable",
                ]
            ].to_dict(orient="records"),
        }

        model_metadata = {
            "project_name": PROJECT_NAME,
            "task_type": TASK_TYPE,
            "selected_candidate_id": winner["_candidate_id"],
            "library_source": winner["library_source"],
            "model_name": winner["model_name"],
            "rank_score": float(winner["rank_score"]),
            "final_rank": int(winner["final_rank"]),
            "holdout_primary_metric": float(winner["holdout_primary_metric"]),
            "holdout_secondary_metric": float(winner["holdout_secondary_metric"]),
            "holdout_tertiary_metric": float(winner["holdout_tertiary_metric"])
            if pd.notna(winner["holdout_tertiary_metric"])
            else None,
            "calibration_metric": float(winner["calibration_metric"])
            if pd.notna(winner["calibration_metric"])
            else None,
            "final_model_path": str(final_model_path),
            "retraining_trigger_policy": {
                "drift_trigger": "Retrain when >25% of monitored features have PSI > 0.20.",
                "quality_trigger": "Retrain when expected business cost per 1000 rises by >=20% vs holdout baseline.",
                "cadence_trigger": "Retrain at least every 30 days if enough recent labeled data is available.",
            },
        }

        model_selection_report_path = output_dir / "model_selection_report.json"
        model_metadata_path = output_dir / "model_registry" / "model_metadata.json"

        write_json(model_selection_report_path, report_payload)
        write_json(model_metadata_path, model_metadata)

        safe_log_artifact(leaderboard_path, True)
        safe_log_artifact(model_selection_report_path, True)
        safe_log_artifact(model_metadata_path, True)
        safe_log_metrics(
            {
                "winning_rank_score": float(winner["rank_score"]),
                "winning_cost_per_1000": float(winner["holdout_primary_metric"]),
                "winning_pr_auc": float(winner["holdout_secondary_metric"]),
                "winning_roc_auc": float(winner["holdout_tertiary_metric"])
                if pd.notna(winner["holdout_tertiary_metric"])
                else np.nan,
            },
            True,
        )

        stability_path = output_dir / "reports" / "top3_seed_stability.csv"
        if args.run_stability_check:
            deployable_top3 = leaderboard[
                leaderboard["_candidate_id"].isin(artifact_map.keys())
            ].sort_values("rank_score", ascending=False).head(3)
            stability_df = run_seed_stability_check(
                top_candidates=deployable_top3,
                seeds=parse_seed_list(args.stability_seeds),
                X_train=splits.X_train,
                y_train=splits.y_train,
                X_valid=splits.X_valid,
                y_valid=splits.y_valid,
                X_test=splits.X_test,
                y_test=splits.y_test,
                fp_cost=args.fp_cost,
                fn_cost=args.fn_cost,
                flaml_time_budget=args.flaml_time_budget,
                artifacts_dir=output_dir,
            )
            stability_df.to_csv(stability_path, index=False)
            safe_log_artifact(stability_path, True)
        else:
            stability_df = pd.DataFrame()

    return {
        "leaderboard": leaderboard,
        "winner": winner,
        "lazy_ranked": lazy_ranked,
        "lazy_top3": lazy_top3,
        "manual_diagnostics": manual_diag,
        "flaml_search_summary": flaml_search_summary,
        "flaml_best_config": flaml_config,
        "pycaret_tables": pycaret_tables,
        "stability": stability_df,
    }


def main() -> None:
    args = parse_args()
    outputs = run_full_training_pipeline(args)

    leaderboard_path = Path(args.output_dir) / "leaderboard_e2e.csv"
    model_selection_path = Path(args.output_dir) / "model_selection_report.json"
    final_model_path = Path(args.output_dir) / "model_registry" / "final_model.joblib"

    print(f"Training complete. Leaderboard saved to: {leaderboard_path}")
    print(f"Model selection report saved to: {model_selection_path}")
    print(f"Final deployable model saved to: {final_model_path}")

    winner = outputs["winner"]
    print(
        "Winner -> "
        f"{winner['library_source']}::{winner['model_name']} "
        f"| rank_score={winner['rank_score']:.2f} "
        f"| cost_per_1000={winner['holdout_primary_metric']:.2f}"
    )


if __name__ == "__main__":
    main()
