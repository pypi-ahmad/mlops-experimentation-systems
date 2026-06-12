from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.infer import InferenceRuntime, load_runtime

app = FastAPI(
    title="Churn Risk Inference API",
    description="Single and batch prediction endpoints for churn risk scoring.",
    version="0.1.0",
)

runtime: InferenceRuntime | None = None


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    customerID: str | None = Field(default=None, description="Optional customer identifier")
    gender: str
    SeniorCitizen: int | str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float | None = None


class BatchRequest(BaseModel):
    records: list[PredictionRecord]


@app.on_event("startup")
def startup_event() -> None:
    global runtime
    runtime = load_runtime()


@app.get("/health")
def health() -> dict[str, Any]:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model runtime is not loaded.")

    return {
        "status": "ok",
        "model_source": runtime.model_source,
        "model_name": runtime.model_name,
        "threshold": runtime.threshold,
    }


@app.post("/predict/single")
def predict_single(record: PredictionRecord) -> dict[str, Any]:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model runtime is not loaded.")

    prediction = runtime.predict_records([record.model_dump()])[0]
    return {
        "input_count": 1,
        "prediction": prediction,
    }


@app.post("/predict/batch")
def predict_batch(batch: BatchRequest) -> dict[str, Any]:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model runtime is not loaded.")

    records = [row.model_dump() for row in batch.records]
    predictions = runtime.predict_records(records)
    return {
        "input_count": len(records),
        "predictions": predictions,
    }
