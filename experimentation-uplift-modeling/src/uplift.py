from __future__ import annotations

import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_NAME = "experimentation-uplift-modeling"
RANDOM_STATE = 42


@dataclass
class TLearnerModel:
    model_treatment: Any
    model_control: Any
    model_name: str
    library_source: str
    interpretability_note: str
    preprocessor: ColumnTransformer | None = None
    feature_cols: list[str] | None = None

    def predict_uplift(self, x_or_df: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(x_or_df, pd.DataFrame):
            if self.preprocessor is None or self.feature_cols is None:
                raise ValueError("DataFrame inference requires fitted preprocessor and feature_cols.")
            x_matrix = self.preprocessor.transform(x_or_df[self.feature_cols])
        else:
            x_matrix = np.asarray(x_or_df)

        p1 = predict_positive_class(self.model_treatment, x_matrix)
        p0 = predict_positive_class(self.model_control, x_matrix)
        return p1 - p0


@dataclass
class XLearnerModel:
    mu_treatment: Any
    mu_control: Any
    tau_treatment: Any
    tau_control: Any
    propensity_model: Any
    model_name: str
    library_source: str
    interpretability_note: str

    def predict_uplift(self, x: np.ndarray) -> np.ndarray:
        tau_t = np.asarray(self.tau_treatment.predict(x), dtype=float)
        tau_c = np.asarray(self.tau_control.predict(x), dtype=float)
        e_x = np.asarray(predict_positive_class(self.propensity_model, x), dtype=float)
        return e_x * tau_c + (1.0 - e_x) * tau_t


def _safe_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _as_binary_treatment(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "b", "treatment", "test", "variant", "yes", "true", "blue"])
        .astype(int)
    )


def load_experiment_data(project_root: str | Path, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(random_state)
    data_path = Path(project_root) / "data" / "raw" / "pharma" / "pharma_ab_test_data.csv"

    if data_path.exists():
        raw = pd.read_csv(data_path)

        df = raw.copy()
        df["treatment"] = _as_binary_treatment(df["group"])
        df["converted"] = df["converted"].astype(int)
        df["is_returning_user"] = df["previous_app_user"].astype(int)
        df["user_type"] = np.where(df["is_returning_user"] == 1, "returning", "new")

        visit_date = pd.to_datetime(df["visit_date"], errors="coerce")
        visit_time = pd.to_datetime(df["visit_time"], format="%H:%M", errors="coerce")
        df["visit_weekday"] = visit_date.dt.day_name().fillna("Unknown")
        df["visit_hour"] = visit_time.dt.hour.fillna(12).astype(int)

        value_multiplier = 1 + 0.05 * df["is_returning_user"] + 0.08 * (df["device_type"].astype(str) == "iOS")
        order_value = rng.lognormal(mean=3.75, sigma=0.45, size=len(df)) * value_multiplier
        df["revenue"] = np.where(df["converted"] == 1, order_value, 0.0)

        feature_columns = [
            "age_group",
            "gender",
            "device_type",
            "previous_app_user",
            "previous_product_buyer",
            "visit_weekday",
            "visit_hour",
            "user_type",
        ]

        curated = df[["user_id", "treatment", "converted", "revenue", *feature_columns]].copy()
        curated = engineer_features(curated)
        return curated, "kaggle: storytellerman/pharma-ab-test-packaging-impact-in-mobile-app"

    synthetic = generate_synthetic_experiment_data(n_samples=5000, random_state=random_state)
    synthetic = engineer_features(synthetic)
    return synthetic, "synthetic-fallback"


def generate_synthetic_experiment_data(
    n_samples: int = 5000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age_group = rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], p=[0.18, 0.28, 0.24, 0.18, 0.12], size=n_samples)
    device_type = rng.choice(["Android", "iOS", "Web"], p=[0.50, 0.35, 0.15], size=n_samples)
    gender = rng.choice(["F", "M"], p=[0.52, 0.48], size=n_samples)
    previous_app_user = rng.binomial(1, 0.58, size=n_samples)
    previous_product_buyer = rng.binomial(1, 0.42, size=n_samples)
    visit_weekday = rng.choice(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        size=n_samples,
    )
    visit_hour = rng.integers(7, 23, size=n_samples)

    treatment = rng.binomial(1, 0.5, size=n_samples)

    baseline = (
        -2.35
        + 0.28 * (device_type == "iOS")
        + 0.16 * previous_app_user
        + 0.25 * previous_product_buyer
        + 0.08 * ((age_group == "25-34") | (age_group == "35-44"))
        + 0.06 * ((visit_hour >= 19) & (visit_hour <= 22))
    )
    base_prob = 1 / (1 + np.exp(-baseline))

    hetero_uplift = (
        0.010
        + 0.020 * ((device_type == "iOS") & (previous_app_user == 0))
        + 0.015 * ((age_group == "25-34") & (previous_product_buyer == 1))
        - 0.010 * ((device_type == "Web") & (previous_app_user == 1))
    )

    final_prob = np.clip(base_prob + treatment * hetero_uplift, 0.001, 0.999)
    converted = rng.binomial(1, final_prob)

    base_order = rng.lognormal(mean=3.7, sigma=0.45, size=n_samples)
    uplift_value = 1 + 0.09 * ((device_type == "iOS") & (previous_app_user == 0))
    revenue = np.where(converted == 1, base_order * uplift_value * (1 + 0.04 * treatment), 0.0)

    user_type = np.where(previous_app_user == 1, "returning", "new")

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n_samples + 1),
            "treatment": treatment,
            "converted": converted,
            "revenue": revenue,
            "user_type": user_type,
            "age_group": age_group,
            "gender": gender,
            "device_type": device_type,
            "previous_app_user": previous_app_user,
            "previous_product_buyer": previous_product_buyer,
            "visit_weekday": visit_weekday,
            "visit_hour": visit_hour,
        }
    )


def load_raw_dataset(project_root: str | Path) -> pd.DataFrame:
    data_path = Path(project_root) / "data" / "raw" / "pharma" / "pharma_ab_test_data.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return pd.DataFrame()


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "n_missing": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "n_unique": int(series.nunique(dropna=True)),
                "sample_values": ", ".join(series.dropna().astype(str).head(3).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def detect_leakage_columns(columns: list[str]) -> pd.DataFrame:
    leakage_tokens = ["purchase", "added_to_cart", "scrolled", "time_on_page", "datetime"]
    rows: list[dict[str, Any]] = []
    for col in columns:
        risk = any(token in col.lower() for token in leakage_tokens)
        rows.append(
            {
                "column": col,
                "is_potential_leakage": bool(risk),
                "reason": "post-treatment behavioral signal" if risk else "kept as pre-treatment / neutral",
            }
        )
    return pd.DataFrame(rows)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "age_group" in out.columns:
        out["age_group"] = out["age_group"].astype(str).str.replace("–", "-", regex=False)
        out["age_group_code"] = out["age_group"].map(
            {
                "18-24": 1,
                "25-34": 2,
                "35-44": 3,
                "45-54": 4,
                "55+": 5,
            }
        ).fillna(0).astype(int)

    if "visit_weekday" in out.columns:
        out["is_weekend"] = out["visit_weekday"].isin(["Saturday", "Sunday"]).astype(int)

    if "device_type" in out.columns:
        out["is_ios"] = (out["device_type"].astype(str) == "iOS").astype(int)

    if "visit_hour" in out.columns:
        hour = out["visit_hour"].astype(float)
        out["visit_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["visit_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    if {"previous_app_user", "is_ios"}.issubset(out.columns):
        out["returning_ios"] = (out["previous_app_user"].astype(int) * out["is_ios"].astype(int)).astype(int)

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"user_id", "treatment", "converted", "revenue"}
    return [c for c in df.columns if c not in excluded]


def split_train_valid_holdout(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "converted",
    valid_size: float = 0.20,
    holdout_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if valid_size <= 0 or holdout_size <= 0 or (valid_size + holdout_size) >= 1:
        raise ValueError("Require 0 < valid_size, holdout_size and valid_size + holdout_size < 1.")

    strata = df[treatment_col].astype(str) + "_" + df[outcome_col].astype(str)

    splitter_holdout = StratifiedShuffleSplit(n_splits=1, test_size=holdout_size, random_state=random_state)
    train_valid_idx, holdout_idx = next(splitter_holdout.split(df, strata))

    train_valid_df = df.iloc[train_valid_idx].reset_index(drop=True)
    holdout_df = df.iloc[holdout_idx].reset_index(drop=True)

    valid_ratio_within_train_valid = valid_size / (1.0 - holdout_size)
    strata_train_valid = (
        train_valid_df[treatment_col].astype(str) + "_" + train_valid_df[outcome_col].astype(str)
    )

    splitter_valid = StratifiedShuffleSplit(
        n_splits=1,
        test_size=valid_ratio_within_train_valid,
        random_state=random_state + 1,
    )
    train_idx, valid_idx = next(splitter_valid.split(train_valid_df, strata_train_valid))

    train_df = train_valid_df.iloc[train_idx].reset_index(drop=True)
    valid_df = train_valid_df.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df, holdout_df


def build_preprocessor(train_df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    categorical_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _safe_ohe()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    preprocessor.fit(train_df[feature_cols])
    return preprocessor


def transform_features(preprocessor: ColumnTransformer, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return np.asarray(preprocessor.transform(frame[feature_cols]))


def predict_positive_class(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(x))
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1]
        return probs.ravel()

    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-z))

    pred = np.asarray(model.predict(x), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def estimator_from_family_name(name: str, random_state: int = RANDOM_STATE) -> Any:
    lname = name.lower().replace(" ", "")

    if "logisticregression" in lname or lname == "lr":
        return LogisticRegression(max_iter=2500, n_jobs=-1, random_state=random_state)
    if "randomforest" in lname or lname == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1, min_samples_leaf=5)
    if "extratrees" in lname or lname == "et":
        return ExtraTreesClassifier(n_estimators=350, random_state=random_state, n_jobs=-1, min_samples_leaf=3)
    if "gradientboosting" in lname:
        return GradientBoostingClassifier(random_state=random_state)
    if "adaboost" in lname:
        return AdaBoostClassifier(random_state=random_state)
    if "decisiontree" in lname:
        return DecisionTreeClassifier(max_depth=7, random_state=random_state, min_samples_leaf=8)
    if "ridgeclassifier" in lname:
        return RidgeClassifier(random_state=random_state)
    if "lineardiscriminantanalysis" in lname:
        return LinearDiscriminantAnalysis()
    if "quadraticdiscriminantanalysis" in lname:
        return QuadraticDiscriminantAnalysis(reg_param=0.1)
    if "gaussiannb" in lname:
        return GaussianNB()
    if "kneighborsclassifier" in lname or "knn" in lname:
        return KNeighborsClassifier(n_neighbors=25, weights="distance")

    if "lightgbm" in lname or "lgbm" in lname:
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(random_state=random_state, n_estimators=350, learning_rate=0.05)
        except Exception:
            return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)

    if "xgb" in lname or "xgboost" in lname:
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        except Exception:
            return GradientBoostingClassifier(random_state=random_state)

    return LogisticRegression(max_iter=2500, n_jobs=-1, random_state=random_state)


def _supports_calibration(model: Any) -> bool:
    return hasattr(model, "predict_proba") or hasattr(model, "decision_function")


def fit_t_learner(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    family_name: str,
    library_source: str,
    model_name: str,
    random_state: int = RANDOM_STATE,
    calibrate: bool = False,
    preprocessor: ColumnTransformer | None = None,
) -> TLearnerModel:
    local_preprocessor = preprocessor or build_preprocessor(train_df, feature_cols)
    x_train = transform_features(local_preprocessor, train_df, feature_cols)
    y_train = train_df["converted"].to_numpy(dtype=int)
    w_train = train_df["treatment"].to_numpy(dtype=int)

    base_model_t = estimator_from_family_name(family_name, random_state=random_state)
    base_model_c = estimator_from_family_name(family_name, random_state=random_state + 1)

    x_t, y_t = x_train[w_train == 1], y_train[w_train == 1]
    x_c, y_c = x_train[w_train == 0], y_train[w_train == 0]

    model_t: Any = clone(base_model_t)
    model_c: Any = clone(base_model_c)

    if calibrate and _supports_calibration(base_model_t) and _supports_calibration(base_model_c):
        try:
            model_t = CalibratedClassifierCV(base_model_t, method="sigmoid", cv=3)
            model_c = CalibratedClassifierCV(base_model_c, method="sigmoid", cv=3)
        except Exception:
            model_t = clone(base_model_t)
            model_c = clone(base_model_c)

    model_t.fit(x_t, y_t)
    model_c.fit(x_c, y_c)

    note = (
        "High: calibrated two-model uplift with manual preprocessing and policy cutoff tuning."
        if calibrate
        else "Moderate: two-model uplift without explicit probability calibration."
    )

    return TLearnerModel(
        model_treatment=model_t,
        model_control=model_c,
        model_name=model_name,
        library_source=library_source,
        interpretability_note=note,
        preprocessor=local_preprocessor,
        feature_cols=feature_cols,
    )


def cumulative_uplift_curve(y_true: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-uplift_score)
    y_sorted = y_true[order]
    w_sorted = treatment[order]

    p_t = np.clip(np.mean(treatment), 1e-6, 1 - 1e-6)
    incremental = (w_sorted * y_sorted / p_t) - ((1 - w_sorted) * y_sorted / (1 - p_t))

    cumulative = np.cumsum(incremental)
    population_share = np.arange(1, len(y_true) + 1) / len(y_true)
    return population_share, cumulative


def uplift_metrics(y_true: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> dict[str, float]:
    x_axis, gains = cumulative_uplift_curve(y_true, treatment, uplift_score)
    auuc = float(np.trapz(gains, x_axis) / len(y_true))

    random_baseline = x_axis * gains[-1]
    qini = float(np.trapz(gains - random_baseline, x_axis) / len(y_true))

    top_n = max(1, int(0.1 * len(y_true)))
    top_idx = np.argsort(-uplift_score)[:top_n]
    y_top = y_true[top_idx]
    w_top = treatment[top_idx]
    p_t = np.clip(np.mean(treatment), 1e-6, 1 - 1e-6)
    uplift_top_decile = float(np.mean((w_top * y_top / p_t) - ((1 - w_top) * y_top / (1 - p_t))))

    return {
        "auuc": auuc,
        "qini": qini,
        "uplift_at_top_decile": uplift_top_decile,
    }


def bootstrap_primary_metric(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
    n_bootstrap: int = 250,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    values: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values.append(uplift_metrics(y_true[idx], treatment[idx], uplift_score[idx])["auuc"])

    return float(np.mean(values)), float(np.std(values))


def evaluate_runtime_metrics(model: TLearnerModel, holdout_df: pd.DataFrame) -> tuple[float, float]:
    start = time.perf_counter()
    _ = model.predict_uplift(holdout_df)
    elapsed = time.perf_counter() - start
    latency_ms = 1000.0 * elapsed / max(len(holdout_df), 1)

    size_mb = float("nan")
    try:
        size_mb = len(pickle.dumps(model)) / (1024 * 1024)
    except Exception:
        pass

    return float(latency_ms), float(size_mb)


def policy_value_ips(y: np.ndarray, w: np.ndarray, policy: np.ndarray) -> float:
    p_t = np.clip(np.mean(w), 1e-6, 1 - 1e-6)
    treated_component = policy * (w * y / p_t)
    control_component = (1 - policy) * ((1 - w) * y / (1 - p_t))
    return float(np.mean(treated_component + control_component))


def simulate_policies(
    y: np.ndarray,
    w: np.ndarray,
    uplift_score: np.ndarray,
    conversion_value: float = 120.0,
    top_fraction: float = 0.30,
) -> dict[str, float]:
    n = len(y)
    no_treat = np.zeros(n, dtype=int)
    blanket = np.ones(n, dtype=int)

    k = max(1, int(top_fraction * n))
    targeted = np.zeros(n, dtype=int)
    targeted[np.argsort(-uplift_score)[:k]] = 1

    value_none = policy_value_ips(y, w, no_treat)
    value_blanket = policy_value_ips(y, w, blanket)
    value_targeted = policy_value_ips(y, w, targeted)

    return {
        "expected_incremental_conversions_blanket": (value_blanket - value_none) * n,
        "expected_incremental_conversions_targeted": (value_targeted - value_none) * n,
        "expected_incremental_value_blanket": (value_blanket - value_none) * n * conversion_value,
        "expected_incremental_value_targeted": (value_targeted - value_none) * n * conversion_value,
        "target_fraction": float(top_fraction),
    }


def optimize_policy_fraction(
    model: TLearnerModel,
    valid_df: pd.DataFrame,
    conversion_value: float = 120.0,
    fraction_grid: list[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    fractions = fraction_grid or [0.10, 0.20, 0.30, 0.40, 0.50]

    y_valid = valid_df["converted"].to_numpy(dtype=int)
    w_valid = valid_df["treatment"].to_numpy(dtype=int)
    uplift_valid = model.predict_uplift(valid_df)

    rows: list[dict[str, float]] = []
    for frac in fractions:
        policy = simulate_policies(
            y=y_valid,
            w=w_valid,
            uplift_score=uplift_valid,
            conversion_value=conversion_value,
            top_fraction=float(frac),
        )
        rows.append(
            {
                "target_fraction": float(frac),
                "expected_incremental_value_targeted": float(policy["expected_incremental_value_targeted"]),
                "expected_incremental_conversions_targeted": float(policy["expected_incremental_conversions_targeted"]),
            }
        )

    table = pd.DataFrame(rows).sort_values("expected_incremental_value_targeted", ascending=False).reset_index(drop=True)
    best_fraction = float(table.loc[0, "target_fraction"])
    return best_fraction, table


def calibration_metric_for_t_learner(model: TLearnerModel, holdout_df: pd.DataFrame) -> float:
    treated = holdout_df[holdout_df["treatment"] == 1]
    control = holdout_df[holdout_df["treatment"] == 0]

    if treated.empty or control.empty:
        return float("nan")

    y_t = treated["converted"].to_numpy(dtype=int)
    y_c = control["converted"].to_numpy(dtype=int)

    if model.library_source == "pycaret":
        if model.feature_cols is None:
            return float("nan")
        prob_t = np.clip(
            predict_positive_class(model.model_treatment, treated[model.feature_cols]),
            1e-6,
            1 - 1e-6,
        )
        prob_c = np.clip(
            predict_positive_class(model.model_control, control[model.feature_cols]),
            1e-6,
            1 - 1e-6,
        )
    else:
        x_t = transform_features(model.preprocessor, treated, model.feature_cols or [])
        x_c = transform_features(model.preprocessor, control, model.feature_cols or [])
        prob_t = np.clip(predict_positive_class(model.model_treatment, x_t), 1e-6, 1 - 1e-6)
        prob_c = np.clip(predict_positive_class(model.model_control, x_c), 1e-6, 1 - 1e-6)

    brier_t = brier_score_loss(y_t, prob_t)
    brier_c = brier_score_loss(y_c, prob_c)
    return float((brier_t + brier_c) / 2.0)


def evaluate_candidate(
    model: TLearnerModel,
    holdout_df: pd.DataFrame,
    candidate_key: str,
    selected_fraction: float = 0.30,
    conversion_value: float = 120.0,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], np.ndarray, dict[str, float]]:
    y_holdout = holdout_df["converted"].to_numpy(dtype=int)
    w_holdout = holdout_df["treatment"].to_numpy(dtype=int)

    uplift_score = model.predict_uplift(holdout_df)
    metrics = uplift_metrics(y_holdout, w_holdout, uplift_score)
    cv_mean, cv_std = bootstrap_primary_metric(
        y_true=y_holdout,
        treatment=w_holdout,
        uplift_score=uplift_score,
        random_state=random_state,
    )

    infer_latency_ms, model_size_mb = evaluate_runtime_metrics(model, holdout_df)
    calibration_metric = calibration_metric_for_t_learner(model, holdout_df)

    policy = simulate_policies(
        y=y_holdout,
        w=w_holdout,
        uplift_score=uplift_score,
        conversion_value=conversion_value,
        top_fraction=selected_fraction,
    )

    row = {
        "project_name": PROJECT_NAME,
        "task_type": "uplift_modeling",
        "library_source": model.library_source,
        "model_name": model.model_name,
        "candidate_key": candidate_key,
        "cv_metric_mean": cv_mean,
        "cv_metric_std": cv_std,
        "holdout_primary_metric": metrics["auuc"],
        "holdout_secondary_metric": metrics["qini"],
        "holdout_tertiary_metric": metrics["uplift_at_top_decile"],
        "calibration_metric": calibration_metric,
        "train_time_sec": float("nan"),
        "infer_latency_ms": infer_latency_ms,
        "model_size_mb": model_size_mb,
        "interpretability_note": model.interpretability_note,
        "selected_target_fraction": float(selected_fraction),
        "policy_incremental_value": float(policy["expected_incremental_value_targeted"]),
    }

    return row, uplift_score, policy


def _selected_lazy_estimators() -> list[type[Any]]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    candidates: list[type[Any]] = [
        LogisticRegression,
        RidgeClassifier,
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
        DecisionTreeClassifier,
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
        GaussianNB,
        KNeighborsClassifier,
    ]

    try:
        from lightgbm import LGBMClassifier

        candidates.append(LGBMClassifier)
    except Exception:
        pass

    try:
        from xgboost import XGBClassifier

        candidates.append(XGBClassifier)
    except Exception:
        pass

    return candidates


def _prepare_lazy_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index()
    if "Model" in out.columns:
        out = out.rename(columns={"Model": "model_family"})
    elif "index" in out.columns:
        out = out.rename(columns={"index": "model_family"})
    elif out.columns.size > 0:
        out = out.rename(columns={out.columns[0]: "model_family"})

    for c in out.columns:
        if c != "model_family":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def run_lazypredict_discovery_lab(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, list[str], dict[str, pd.DataFrame]]:
    from lazypredict.Supervised import LazyClassifier

    preprocessor = build_preprocessor(train_df, feature_cols)

    def _run_one_arm(arm_value: int) -> pd.DataFrame:
        arm_train = train_df[train_df["treatment"] == arm_value]
        arm_valid = valid_df[valid_df["treatment"] == arm_value]

        x_train = transform_features(preprocessor, arm_train, feature_cols)
        y_train = arm_train["converted"].to_numpy(dtype=int)

        x_valid = transform_features(preprocessor, arm_valid, feature_cols)
        y_valid = arm_valid["converted"].to_numpy(dtype=int)

        lazy = LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            predictions=False,
            custom_metric=None,
            classifiers=_selected_lazy_estimators(),
            random_state=random_state + arm_value,
        )
        leaderboard, _ = lazy.fit(x_train, x_valid, y_train, y_valid)
        return _prepare_lazy_table(leaderboard)

    table_t = _run_one_arm(1)
    table_c = _run_one_arm(0)

    merged = table_t.merge(table_c, on="model_family", suffixes=("_treat", "_ctrl"), how="inner")

    merged["balanced_accuracy_mean"] = merged[["Balanced Accuracy_treat", "Balanced Accuracy_ctrl"]].mean(axis=1)
    roc_cols = [c for c in ["ROC AUC_treat", "ROC AUC_ctrl"] if c in merged.columns]
    merged["roc_auc_mean"] = merged[roc_cols].mean(axis=1) if roc_cols else np.nan

    time_cols = [c for c in ["Time Taken_treat", "Time Taken_ctrl"] if c in merged.columns]
    merged["time_taken_mean"] = merged[time_cols].mean(axis=1) if time_cols else np.nan

    excluded = {
        "DummyClassifier",
        "LabelPropagation",
        "LabelSpreading",
        "PassiveAggressiveClassifier",
        "Perceptron",
        "SGDClassifier",
        "NearestCentroid",
    }

    eligible = merged[~merged["model_family"].isin(excluded)].copy()
    eligibility_cutoff = float(eligible["balanced_accuracy_mean"].quantile(0.40)) if not eligible.empty else -np.inf
    eligible = eligible[eligible["balanced_accuracy_mean"] >= eligibility_cutoff]

    ranked = eligible.sort_values(
        ["balanced_accuracy_mean", "roc_auc_mean", "time_taken_mean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    top3 = ranked["model_family"].head(3).tolist()
    if len(top3) < 3:
        fallback = merged.sort_values("balanced_accuracy_mean", ascending=False)["model_family"].tolist()
        for model_name in fallback:
            if model_name not in top3:
                top3.append(model_name)
            if len(top3) == 3:
                break

    return ranked, top3, {"treatment_arm": table_t, "control_arm": table_c}


def run_lazy_top_family_prototypes(
    top_families: list[str],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
    conversion_value: float = 120.0,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, TLearnerModel]]:
    rows: list[dict[str, Any]] = []
    models: dict[str, TLearnerModel] = {}

    preprocessor = build_preprocessor(train_df, feature_cols)

    for idx, family in enumerate(top_families):
        start = time.perf_counter()
        model = fit_t_learner(
            train_df=train_df,
            feature_cols=feature_cols,
            family_name=family,
            library_source="lazypredict",
            model_name=f"LazyPrototype T-Learner ({family})",
            random_state=random_state + idx,
            calibrate=False,
            preprocessor=preprocessor,
        )
        train_time = time.perf_counter() - start

        row, _, _ = evaluate_candidate(
            model=model,
            holdout_df=holdout_df,
            candidate_key=f"lazy_proto::{family}",
            selected_fraction=0.30,
            conversion_value=conversion_value,
            random_state=random_state + idx,
        )
        row["train_time_sec"] = float(train_time)
        row["interpretability_note"] = "Discovery-only prototype from LazyPredict family ranking."
        rows.append(row)
        models[row["candidate_key"]] = model

    return pd.DataFrame(rows), models


def run_manual_engineering_lab(
    top_families: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
    conversion_value: float = 120.0,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, TLearnerModel], dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    models: dict[str, TLearnerModel] = {}
    tuning_tables: dict[str, pd.DataFrame] = {}

    preprocessor = build_preprocessor(train_df, feature_cols)

    for idx, family in enumerate(top_families):
        start = time.perf_counter()
        model = fit_t_learner(
            train_df=train_df,
            feature_cols=feature_cols,
            family_name=family,
            library_source="manual",
            model_name=f"Manual Calibrated T-Learner ({family})",
            random_state=random_state + idx,
            calibrate=True,
            preprocessor=preprocessor,
        )

        best_fraction, tuning_table = optimize_policy_fraction(
            model=model,
            valid_df=valid_df,
            conversion_value=conversion_value,
        )
        train_time = time.perf_counter() - start

        candidate_key = f"manual::{family}"
        row, _, _ = evaluate_candidate(
            model=model,
            holdout_df=holdout_df,
            candidate_key=candidate_key,
            selected_fraction=best_fraction,
            conversion_value=conversion_value,
            random_state=random_state + idx,
        )
        row["train_time_sec"] = float(train_time)

        rows.append(row)
        models[candidate_key] = model
        tuning_tables[candidate_key] = tuning_table

    return pd.DataFrame(rows), models, tuning_tables


def _fit_flaml_classifier(x: np.ndarray, y: np.ndarray, random_state: int, time_budget: int = 40) -> Any:
    from flaml import AutoML

    automl = AutoML()
    automl.fit(
        X_train=x,
        y_train=y,
        task="classification",
        metric="log_loss",
        time_budget=time_budget,
        estimator_list=["lgbm", "rf", "extra_tree", "xgboost"],
        seed=random_state,
        eval_method="cv",
        n_splits=3,
        log_file_name="",
        verbose=0,
    )
    return automl


def run_flaml_optimization_lab(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
    conversion_value: float = 120.0,
    random_state: int = RANDOM_STATE,
    time_budget: int = 40,
) -> tuple[pd.DataFrame, TLearnerModel, pd.DataFrame, dict[str, Any]]:
    preprocessor = build_preprocessor(train_df, feature_cols)
    x_train = transform_features(preprocessor, train_df, feature_cols)
    y_train = train_df["converted"].to_numpy(dtype=int)
    w_train = train_df["treatment"].to_numpy(dtype=int)

    x_t = x_train[w_train == 1]
    y_t = y_train[w_train == 1]
    x_c = x_train[w_train == 0]
    y_c = y_train[w_train == 0]

    start = time.perf_counter()
    automl_t = _fit_flaml_classifier(x_t, y_t, random_state=random_state, time_budget=time_budget)
    automl_c = _fit_flaml_classifier(x_c, y_c, random_state=random_state + 1, time_budget=time_budget)

    model = TLearnerModel(
        model_treatment=automl_t,
        model_control=automl_c,
        model_name="FLAML Optimized T-Learner",
        library_source="flaml",
        interpretability_note="Moderate: AutoML-selected response learners with explicit search budgets and CV.",
        preprocessor=preprocessor,
        feature_cols=feature_cols,
    )

    best_fraction, tuning_table = optimize_policy_fraction(model, valid_df, conversion_value=conversion_value)
    train_time = time.perf_counter() - start

    row, _, _ = evaluate_candidate(
        model=model,
        holdout_df=holdout_df,
        candidate_key="flaml::t_learner",
        selected_fraction=best_fraction,
        conversion_value=conversion_value,
        random_state=random_state,
    )
    row["train_time_sec"] = float(train_time)

    details = {
        "treatment_best_estimator": getattr(automl_t, "best_estimator", "unknown"),
        "control_best_estimator": getattr(automl_c, "best_estimator", "unknown"),
        "treatment_best_config": getattr(automl_t, "best_config", {}),
        "control_best_config": getattr(automl_c, "best_config", {}),
        "treatment_best_loss": float(getattr(automl_t, "best_loss", np.nan)),
        "control_best_loss": float(getattr(automl_c, "best_loss", np.nan)),
        "time_budget_per_arm_sec": int(time_budget),
    }

    return pd.DataFrame([row]), model, tuning_table, details


def _extract_pycaret_positive_score(pred_obj: Any) -> np.ndarray:
    pred_df = pred_obj.predictions if hasattr(pred_obj, "predictions") else pred_obj
    if not isinstance(pred_df, pd.DataFrame):
        raise ValueError("Unexpected PyCaret prediction object type.")

    if "prediction_score" in pred_df.columns:
        return pred_df["prediction_score"].to_numpy(dtype=float)

    score_cols = [c for c in pred_df.columns if "score" in c.lower()]
    if not score_cols:
        if "prediction_label" in pred_df.columns:
            return pred_df["prediction_label"].astype(float).to_numpy()
        raise ValueError("PyCaret output does not include probability scores.")

    preferred = [
        c
        for c in score_cols
        if c.lower().endswith("_1")
        or c.lower() == "score_1"
        or c.lower().endswith("positive")
    ]
    col = preferred[0] if preferred else score_cols[-1]
    return pred_df[col].to_numpy(dtype=float)


def _fit_pycaret_arm(
    arm_train: pd.DataFrame,
    feature_cols: list[str],
    model_prefix: str,
    random_state: int,
) -> tuple[Any, Any, pd.DataFrame, pd.DataFrame, str]:
    from pycaret.classification import ClassificationExperiment

    exp = ClassificationExperiment(
        target="converted",
        session_id=random_state,
        fold=3,
        preprocess=True,
        n_jobs=1,
        use_gpu=False,
        verbose=False,
    )
    exp.fit(X=arm_train[feature_cols + ["converted"]])

    compare_result = exp.compare_models(
        include=["lr", "rf", "et", "lightgbm", "xgboost"],
        sort="AUC",
        turbo=False,
        errors="ignore",
        verbose=False,
    )
    compare_table = (
        compare_result.leaderboard.copy()
        if hasattr(compare_result, "leaderboard")
        else exp.pull().copy()
    )

    best_pipeline = compare_result.best if hasattr(compare_result, "best") else compare_result
    tune_result = exp.tune_model(best_pipeline, optimize="AUC", verbose=False)
    tune_table = tune_result.metrics.copy() if hasattr(tune_result, "metrics") else exp.pull().copy()
    tuned_pipeline = tune_result.pipeline if hasattr(tune_result, "pipeline") else tune_result

    calibrate_result = exp.calibrate_model(tuned_pipeline, method="sigmoid", verbose=False)
    calibrated_pipeline = (
        calibrate_result.pipeline if hasattr(calibrate_result, "pipeline") else calibrate_result
    )

    finalize_result = exp.finalize_model(calibrated_pipeline)
    final_model = finalize_result.pipeline if hasattr(finalize_result, "pipeline") else finalize_result
    save_name = f"artifacts/models/{model_prefix}_pycaret"
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    saved_path = exp.save_model(final_model, save_name, verbose=False)

    return exp, final_model, compare_table, tune_table, str(saved_path)


def run_pycaret_experiment_lab(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
    conversion_value: float = 120.0,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, TLearnerModel, pd.DataFrame, dict[str, Any]]:
    start = time.perf_counter()

    treated_train = train_df[train_df["treatment"] == 1].reset_index(drop=True)
    control_train = train_df[train_df["treatment"] == 0].reset_index(drop=True)

    exp_t, final_t, compare_t, tune_t, path_t = _fit_pycaret_arm(
        treated_train,
        feature_cols,
        model_prefix="treatment",
        random_state=random_state,
    )
    exp_c, final_c, compare_c, tune_c, path_c = _fit_pycaret_arm(
        control_train,
        feature_cols,
        model_prefix="control",
        random_state=random_state + 1,
    )

    class _PyCaretAdapter:
        def __init__(self, exp_obj: Any, model_obj: Any):
            self.exp_obj = exp_obj
            self.model_obj = model_obj

        def predict_proba(self, x_df: pd.DataFrame) -> np.ndarray:
            pred = self.exp_obj.predict_model(self.model_obj, data=x_df.copy(), raw_score=True, verbose=False)
            pos = _extract_pycaret_positive_score(pred)
            return np.column_stack([1.0 - pos, pos])

    adapter_t = _PyCaretAdapter(exp_t, final_t)
    adapter_c = _PyCaretAdapter(exp_c, final_c)

    class _NoopPreprocessor:
        def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame

    model = TLearnerModel(
        model_treatment=adapter_t,
        model_control=adapter_c,
        model_name="PyCaret Finalized T-Learner",
        library_source="pycaret",
        interpretability_note="Lower: orchestrated AutoML workflow (compare, tune, calibrate, finalize, save).",
        preprocessor=_NoopPreprocessor(),
        feature_cols=feature_cols,
    )

    best_fraction, tuning_table = optimize_policy_fraction(model, valid_df, conversion_value=conversion_value)

    row, _, _ = evaluate_candidate(
        model=model,
        holdout_df=holdout_df,
        candidate_key="pycaret::t_learner",
        selected_fraction=best_fraction,
        conversion_value=conversion_value,
        random_state=random_state,
    )
    row["train_time_sec"] = float(time.perf_counter() - start)

    details = {
        "treatment_compare_top": compare_t.head(5),
        "control_compare_top": compare_c.head(5),
        "treatment_tune_result": tune_t.head(5),
        "control_tune_result": tune_c.head(5),
        "saved_treatment_model": path_t,
        "saved_control_model": path_c,
    }

    return pd.DataFrame([row]), model, tuning_table, details


def build_logistic_baseline_row(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, TLearnerModel]:
    start = time.perf_counter()
    model = fit_t_learner(
        train_df=train_df,
        feature_cols=feature_cols,
        family_name="LogisticRegression",
        library_source="baseline",
        model_name="Baseline Logistic T-Learner",
        random_state=random_state,
        calibrate=False,
    )

    row, _, _ = evaluate_candidate(
        model=model,
        holdout_df=holdout_df,
        candidate_key="baseline::logistic",
        selected_fraction=0.30,
        conversion_value=120.0,
        random_state=random_state,
    )
    row["train_time_sec"] = float(time.perf_counter() - start)
    row["interpretability_note"] = "High: transparent baseline for uplift effect ranking."

    return pd.DataFrame([row]), model


def _normalize_for_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.isna().any() and not clean.isna().all():
        fill_value = clean.min() if higher_is_better else clean.max()
        clean = clean.fillna(fill_value)
    if clean.isna().all() or float(clean.max() - clean.min()) == 0.0:
        return pd.Series(np.ones(len(clean)), index=clean.index)

    scaled = (clean - clean.min()) / (clean.max() - clean.min())
    return scaled if higher_is_better else 1.0 - scaled


def build_unified_leaderboard(rows: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
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

    board = rows.copy()

    for col in required_cols:
        if col not in board.columns:
            board[col] = np.nan

    board["rank_score"] = (
        0.33 * _normalize_for_rank(board["holdout_primary_metric"], True)
        + 0.18 * _normalize_for_rank(board["holdout_secondary_metric"], True)
        + 0.12 * _normalize_for_rank(board["holdout_tertiary_metric"], True)
        + 0.12 * _normalize_for_rank(board["calibration_metric"], False)
        + 0.10 * _normalize_for_rank(board["train_time_sec"], False)
        + 0.08 * _normalize_for_rank(board["infer_latency_ms"], False)
        + 0.07 * _normalize_for_rank(board["model_size_mb"], False)
    )

    board["final_rank"] = board["rank_score"].rank(ascending=False, method="dense").astype(int)
    board = board.sort_values("final_rank").reset_index(drop=True)

    return board[
        [
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
            "candidate_key",
            "selected_target_fraction",
            "policy_incremental_value",
        ]
    ]


def save_unified_leaderboard(leaderboard: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output, index=False)
    return output


def save_t_learner_artifact(model: TLearnerModel, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_treatment": model.model_treatment,
        "model_control": model.model_control,
        "preprocessor": model.preprocessor,
        "feature_cols": model.feature_cols,
        "model_name": model.model_name,
        "library_source": model.library_source,
        "interpretability_note": model.interpretability_note,
    }

    with out.open("wb") as f:
        pickle.dump(payload, f)
    return out


def _train_candidate_by_key(
    candidate_key: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int,
) -> dict[str, float]:
    conversion_value = 120.0

    if candidate_key.startswith("manual::"):
        family = candidate_key.split("::", 1)[1]
        model = fit_t_learner(
            train_df,
            feature_cols,
            family_name=family,
            library_source="manual",
            model_name=f"Manual Calibrated T-Learner ({family})",
            random_state=random_state,
            calibrate=True,
        )
        frac, _ = optimize_policy_fraction(model, valid_df, conversion_value=conversion_value)
        row, _, _ = evaluate_candidate(
            model,
            holdout_df,
            candidate_key=candidate_key,
            selected_fraction=frac,
            conversion_value=conversion_value,
            random_state=random_state,
        )
        row["train_time_sec"] = np.nan
        return row

    if candidate_key.startswith("lazy_proto::"):
        family = candidate_key.split("::", 1)[1]
        model = fit_t_learner(
            train_df,
            feature_cols,
            family_name=family,
            library_source="lazypredict",
            model_name=f"LazyPrototype T-Learner ({family})",
            random_state=random_state,
            calibrate=False,
        )
        row, _, _ = evaluate_candidate(
            model,
            holdout_df,
            candidate_key=candidate_key,
            selected_fraction=0.30,
            conversion_value=conversion_value,
            random_state=random_state,
        )
        row["train_time_sec"] = np.nan
        return row

    if candidate_key == "flaml::t_learner":
        flaml_row, _, _, _ = run_flaml_optimization_lab(
            train_df,
            valid_df,
            holdout_df,
            feature_cols,
            conversion_value=conversion_value,
            random_state=random_state,
            time_budget=20,
        )
        return flaml_row.iloc[0].to_dict()

    if candidate_key == "pycaret::t_learner":
        py_row, _, _, _ = run_pycaret_experiment_lab(
            train_df,
            valid_df,
            holdout_df,
            feature_cols,
            conversion_value=conversion_value,
            random_state=random_state,
        )
        return py_row.iloc[0].to_dict()

    if candidate_key == "baseline::logistic":
        base_row, _ = build_logistic_baseline_row(train_df, holdout_df, feature_cols, random_state=random_state)
        return base_row.iloc[0].to_dict()

    raise ValueError(f"Unsupported candidate key: {candidate_key}")


def rerun_candidates_multiple_seeds(
    df: pd.DataFrame,
    feature_cols: list[str],
    candidate_keys: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        train_df, valid_df, holdout_df = split_train_valid_holdout(df, random_state=seed)
        for key in candidate_keys:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row = _train_candidate_by_key(
                    candidate_key=key,
                    train_df=train_df,
                    valid_df=valid_df,
                    holdout_df=holdout_df,
                    feature_cols=feature_cols,
                    random_state=seed,
                )
            row["seed"] = int(seed)
            rows.append(row)

    out = pd.DataFrame(rows)
    summary = (
        out.groupby("candidate_key", dropna=False)["holdout_primary_metric"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "auuc_mean",
                "std": "auuc_std",
                "min": "auuc_min",
                "max": "auuc_max",
                "count": "n_runs",
            }
        )
    )
    return summary.sort_values("auuc_mean", ascending=False).reset_index(drop=True)


def treatment_control_balance_check(df: pd.DataFrame, treatment_col: str, feature_cols: list[str]) -> pd.DataFrame:
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    rows: list[dict[str, Any]] = []

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            for level in sorted(df[col].dropna().unique()):
                t_rate = float((treated[col] == level).mean())
                c_rate = float((control[col] == level).mean())
                pooled = np.sqrt(max((t_rate * (1 - t_rate) + c_rate * (1 - c_rate)) / 2, 1e-9))
                smd = (t_rate - c_rate) / pooled
                rows.append(
                    {
                        "feature": f"{col}={level}",
                        "treated_mean": t_rate,
                        "control_mean": c_rate,
                        "std_mean_diff": float(smd),
                    }
                )
        else:
            t_mean = float(treated[col].mean())
            c_mean = float(control[col].mean())
            t_var = float(treated[col].var(ddof=1))
            c_var = float(control[col].var(ddof=1))
            pooled_sd = np.sqrt(max((t_var + c_var) / 2, 1e-9))
            smd = (t_mean - c_mean) / pooled_sd
            rows.append(
                {
                    "feature": col,
                    "treated_mean": t_mean,
                    "control_mean": c_mean,
                    "std_mean_diff": float(smd),
                }
            )

    return pd.DataFrame(rows).sort_values("std_mean_diff", key=np.abs, ascending=False).reset_index(drop=True)


def decision_recommendation(
    ab_p_value: float,
    ab_lift: float,
    ci_low: float,
    ci_high: float,
    blanket_value: float,
    targeted_value: float,
) -> dict[str, str | float]:
    if ab_lift > 0 and ab_p_value < 0.05 and ci_low > 0:
        if targeted_value > blanket_value * 1.05:
            decision = "ship_segment_only"
            rationale = "A/B effect is positive with statistical confidence, and targeted rollout materially exceeds blanket value."
        else:
            decision = "ship_all"
            rationale = "A/B effect is positive and reliable, with limited extra value from targeted-only rollout."
    elif ab_lift > 0 and targeted_value > 0:
        decision = "iterate"
        rationale = "Directional lift exists but uncertainty remains; continue experimentation before full launch."
    else:
        decision = "stop"
        rationale = "No reliable positive incremental effect; avoid broad rollout."

    return {
        "decision": decision,
        "rationale": rationale,
        "ab_ci": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "expected_incremental_value_blanket": float(blanket_value),
        "expected_incremental_value_targeted": float(targeted_value),
    }
