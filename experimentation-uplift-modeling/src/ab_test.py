from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_effectsize,
    proportions_ztest,
)


@dataclass
class AbTestResult:
    n_control: int
    n_treatment: int
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass
class PowerCheck:
    baseline_rate: float
    observed_lift: float
    alpha: float
    target_power: float
    required_n_per_group: int
    observed_power: float
    is_sample_adequate: bool


def encode_treatment(series: pd.Series) -> pd.Series:
    """Convert treatment labels to binary (0=control, 1=treatment)."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(pd.Series(series).dropna().unique().tolist())
        if unique_values.issubset({0, 1}):
            return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    treatment_tokens = {"1", "b", "treatment", "test", "variant", "yes", "true"}
    return normalized.isin(treatment_tokens).astype(int)


def estimate_ab_test(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> AbTestResult:
    data = df[[treatment_col, outcome_col]].dropna().copy()
    data[treatment_col] = encode_treatment(data[treatment_col])
    data[outcome_col] = data[outcome_col].astype(int)

    control = data[data[treatment_col] == 0]
    treatment = data[data[treatment_col] == 1]

    n_control = int(control.shape[0])
    n_treatment = int(treatment.shape[0])

    control_success = int(control[outcome_col].sum())
    treatment_success = int(treatment[outcome_col].sum())

    control_rate = control_success / n_control
    treatment_rate = treatment_success / n_treatment

    absolute_lift = treatment_rate - control_rate
    relative_lift = absolute_lift / control_rate if control_rate > 0 else np.nan

    _, p_value = proportions_ztest(
        count=[treatment_success, control_success],
        nobs=[n_treatment, n_control],
        alternative="two-sided",
    )

    ci_low, ci_high = confint_proportions_2indep(
        count1=treatment_success,
        nobs1=n_treatment,
        count2=control_success,
        nobs2=n_control,
        method="wald",
        compare="diff",
    )

    return AbTestResult(
        n_control=n_control,
        n_treatment=n_treatment,
        control_rate=float(control_rate),
        treatment_rate=float(treatment_rate),
        absolute_lift=float(absolute_lift),
        relative_lift=float(relative_lift),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
    )


def randomization_diagnostics(
    df: pd.DataFrame,
    treatment_col: str,
    numeric_cols: Iterable[str] | None = None,
    categorical_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    data = df.copy()
    data[treatment_col] = encode_treatment(data[treatment_col])

    diagnostics: list[dict[str, float | str]] = []

    for col in numeric_cols or []:
        subset = data[[treatment_col, col]].dropna()
        grp0 = subset[subset[treatment_col] == 0][col].astype(float)
        grp1 = subset[subset[treatment_col] == 1][col].astype(float)
        if grp0.empty or grp1.empty:
            continue
        _, p_value = stats.ttest_ind(grp1, grp0, equal_var=False)
        diagnostics.append(
            {
                "feature": col,
                "test": "welch_ttest",
                "control_mean": float(grp0.mean()),
                "treatment_mean": float(grp1.mean()),
                "difference": float(grp1.mean() - grp0.mean()),
                "p_value": float(p_value),
            }
        )

    for col in categorical_cols or []:
        subset = data[[treatment_col, col]].dropna()
        if subset.empty:
            continue
        contingency = pd.crosstab(subset[col], subset[treatment_col])
        if contingency.shape[1] < 2:
            continue
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        diagnostics.append(
            {
                "feature": col,
                "test": "chi_square",
                "control_mean": np.nan,
                "treatment_mean": np.nan,
                "difference": float(chi2),
                "p_value": float(p_value),
            }
        )

    return pd.DataFrame(diagnostics).sort_values("p_value", ascending=True)


def practical_significance(absolute_lift: float, min_detectable_lift: float = 0.01) -> bool:
    return bool(abs(absolute_lift) >= min_detectable_lift)


def power_and_sample_size_check(
    baseline_rate: float,
    observed_lift: float,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> PowerCheck:
    baseline_rate = float(np.clip(baseline_rate, 1e-6, 1 - 1e-6))
    treatment_rate = float(np.clip(baseline_rate + observed_lift, 1e-6, 1 - 1e-6))

    effect_size = proportion_effectsize(treatment_rate, baseline_rate)
    power_tool = NormalIndPower()

    required_n = power_tool.solve_power(
        effect_size=effect_size,
        power=target_power,
        alpha=alpha,
        ratio=1.0,
        alternative="two-sided",
    )

    observed_power = power_tool.solve_power(
        effect_size=effect_size,
        nobs1=min(n_control, n_treatment),
        alpha=alpha,
        ratio=max(n_treatment, 1) / max(n_control, 1),
        alternative="two-sided",
    )

    required_n_per_group = int(np.ceil(required_n))
    is_sample_adequate = n_control >= required_n_per_group and n_treatment >= required_n_per_group

    return PowerCheck(
        baseline_rate=baseline_rate,
        observed_lift=observed_lift,
        alpha=alpha,
        target_power=target_power,
        required_n_per_group=required_n_per_group,
        observed_power=float(observed_power),
        is_sample_adequate=bool(is_sample_adequate),
    )


def segment_effects(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    segment_cols: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []

    for segment_col in segment_cols:
        for segment_value, frame in df.groupby(segment_col, dropna=False):
            if frame[treatment_col].nunique(dropna=True) < 2:
                continue
            result = estimate_ab_test(frame, treatment_col=treatment_col, outcome_col=outcome_col)
            rows.append(
                {
                    "segment": segment_col,
                    "segment_value": str(segment_value),
                    "n_control": result.n_control,
                    "n_treatment": result.n_treatment,
                    "control_rate": result.control_rate,
                    "treatment_rate": result.treatment_rate,
                    "absolute_lift": result.absolute_lift,
                    "relative_lift": result.relative_lift,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "p_value": result.p_value,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "segment",
                "segment_value",
                "n_control",
                "n_treatment",
                "control_rate",
                "treatment_rate",
                "absolute_lift",
                "relative_lift",
                "ci_low",
                "ci_high",
                "p_value",
            ]
        )

    return pd.DataFrame(rows).sort_values(["segment", "p_value"], ascending=[True, True])
