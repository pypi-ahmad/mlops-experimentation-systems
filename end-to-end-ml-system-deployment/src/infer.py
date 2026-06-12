from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_pipeline import coerce_input_features

DEFAULT_MODEL_PATH = Path("artifacts/model_registry/final_model.joblib")


@dataclass
class PredictionResult:
    churn_probability: float
    predicted_label: int
    threshold: float
    model_source: str
    model_name: str


class InferenceRuntime:
    def __init__(self, model_bundle: dict[str, Any]):
        if not isinstance(model_bundle, dict) or "model" not in model_bundle:
            raise ValueError("Model bundle must be a dict containing at least a 'model' key.")

        self.model = model_bundle["model"]
        self.threshold = float(model_bundle.get("threshold", 0.5))
        self.feature_columns = model_bundle.get("feature_columns")
        self.model_source = str(model_bundle.get("source", "unknown"))
        self.model_name = str(model_bundle.get("model_name", "unknown"))

    def _predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)
            if probabilities.ndim == 2:
                return probabilities[:, 1].astype(float)
            return probabilities.astype(float)

        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(features), dtype=float)

        raise RuntimeError("Loaded model does not support predict or predict_proba.")

    def _prepare_frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(records)
        if self.feature_columns:
            frame = coerce_input_features(frame, self.feature_columns)
        return frame

    def predict_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []

        features = self._prepare_frame(records)
        probabilities = self._predict_proba(features)
        labels = (probabilities >= self.threshold).astype(int)

        results = []
        for idx, probability in enumerate(probabilities):
            results.append(
                {
                    "churn_probability": float(probability),
                    "predicted_label": int(labels[idx]),
                    "threshold": self.threshold,
                    "model_source": self.model_source,
                    "model_name": self.model_name,
                }
            )

        return results


def load_runtime(model_path: str | Path = DEFAULT_MODEL_PATH) -> InferenceRuntime:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run training first with `uv run python -m src.train`."
        )

    bundle = joblib.load(path)
    return InferenceRuntime(bundle)


def load_payload(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload

    raise ValueError("Input payload must be a JSON object or array of objects.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local inference using the selected model registry artifact.")
    parser.add_argument("--input-file", required=True, help="JSON file containing one object or list of objects.")
    parser.add_argument("--output-file", help="Optional output JSON path.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    runtime = load_runtime(args.model_path)
    records = load_payload(Path(args.input_file))
    predictions = runtime.predict_records(records)

    serialized = json.dumps(predictions, indent=2)
    print(serialized)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(serialized)


if __name__ == "__main__":
    main()
