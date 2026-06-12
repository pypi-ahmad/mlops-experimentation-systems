from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Download it first (see README for the Kaggle command)."
        )

    df = pd.read_csv(csv_path)
    return clean_telco_dataframe(df)


def clean_telco_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [col.strip() for col in cleaned.columns]

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(str)

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = (
            cleaned[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
            .astype("Int64")
        )
        cleaned = cleaned[cleaned[TARGET_COLUMN].notna()].copy()
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def prepare_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
) -> tuple[pd.DataFrame, pd.Series | None]:
    work_df = df.copy()

    if target_column in work_df.columns:
        y = work_df[target_column].astype(int)
        X = work_df.drop(columns=[target_column])
    else:
        y = None
        X = work_df

    if id_column in X.columns:
        X = X.drop(columns=[id_column])

    return X, y


def split_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplits:
    X, y = prepare_features_target(df, target_column=target_column)
    if y is None:
        raise ValueError("Target column is required for splitting.")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    valid_fraction = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=valid_fraction,
        random_state=random_state,
        stratify=y_train_full,
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def coerce_input_features(frame: pd.DataFrame, expected_columns: Iterable[str]) -> pd.DataFrame:
    data = frame.copy()

    for column in expected_columns:
        if column not in data.columns:
            data[column] = pd.NA

    filtered = data[list(expected_columns)].copy()

    if "TotalCharges" in filtered.columns:
        filtered["TotalCharges"] = pd.to_numeric(filtered["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in filtered.columns:
        filtered["SeniorCitizen"] = filtered["SeniorCitizen"].astype(str)

    return filtered
