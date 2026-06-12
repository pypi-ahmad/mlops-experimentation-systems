# End-to-End ML System Deployment (Customer Churn Risk)

Compact but complete ML project for customer churn risk scoring with:
- notebook-first experimentation across 4 serious modeling labs
- production-style modular training scripts
- MLflow experiment tracking
- deployable FastAPI inference service
- monitoring-ready drift and batch-scoring artifacts

## Project Goal
Build a reliable churn-risk model pipeline that optimizes business outcomes, not only model AUC.

Primary objective:
- minimize **expected business cost per 1000 predictions**

Secondary objectives:
- maintain predictive quality (PR-AUC / ROC-AUC)
- satisfy operational constraints (latency, model size, retrain effort)

## Dataset
Kaggle dataset: `blastchar/telco-customer-churn`

Download command:
```bash
cd end-to-end-ml-system-deployment
uv run --with kaggle kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
```

Expected file:
- `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Setup

```bash
git clone https://github.com/pypi-ahmad/end-to-end-ml-system-deployment.git
cd end-to-end-ml-system-deployment
```

```bash
cd end-to-end-ml-system-deployment
uv sync
source .venv/bin/activate
```

## Current Workflow (Upgraded)

### Notebook workflow
Notebook: `end_to_end_ml_training.ipynb`

Required section flow implemented:
1. Business Problem and Success Criteria
2. Dataset Access and Data Dictionary
3. Data Cleaning and Leakage Checks
4. Feature Engineering
5. Validation Strategy
6. LazyPredict Discovery Lab
7. Selection of Top 3 Eligible Models
8. Manual Engineering Lab
9. FLAML Optimization Lab
10. PyCaret Experiment Lab
11. Unified Leaderboard and Final Model Ranking
12. Business Recommendation
13. Inference / Deployment Path
14. Monitoring / Drift / Retraining Plan
15. Limitations and Next Steps

### Strict model-family rule
`LazyPredict -> Top 3 Eligible Families -> Manual Engineering`

Manual models are not chosen arbitrarily. They are constrained by the eligible top-3 family shortlist discovered in the LazyPredict lab.

## 4 Serious Modeling Tracks

### 1) LazyPredict Discovery Lab
- runs after full feature matrix + split strategy are defined
- produces ranked benchmark and eligibility filter
- emits `artifacts/reports/lazypredict_top3_eligible.csv`

### 2) Manual Engineering Lab
- manually trains only the top-3 eligible family set
- includes explicit preprocessing, threshold optimization, calibration/error diagnostics
- emits deployable model bundles and diagnostics

### 3) FLAML Optimization Lab
- genuine AutoML optimization with explicit `time_budget`
- uses business-cost-centric search objective (with safe fallback)
- emits search summary and best config artifacts

### 4) PyCaret Experiment Lab
- full orchestration flow: `setup -> compare_models -> tune_model -> calibrate_model -> finalize_model`
- emits compare/tune/calibration tables + finalized artifact

## Unified Leaderboard Logic
Unified leaderboard file:
- `artifacts/leaderboard_e2e.csv`

Tracks included:
- baseline
- top LazyPredict benchmark rows
- manual top-3 implementations
- best FLAML result
- best PyCaret finalized result

Leaderboard schema includes:
- `project_name`
- `task_type`
- `library_source`
- `model_name`
- `cv_metric_mean`
- `cv_metric_std`
- `holdout_primary_metric`
- `holdout_secondary_metric`
- `holdout_tertiary_metric`
- `calibration_metric`
- `train_time_sec`
- `infer_latency_ms`
- `model_size_mb`
- `interpretability_note`
- `rank_score`
- `final_rank`

Ranking objective:
- maximize `rank_score` with business-cost weighted quality and ops constraints

Final winner rule:
- highest rank score among deployable candidates that pass guardrails

## Training / Evaluation / Serving Commands

### Full training pipeline (script)
```bash
uv run python -m src.train --data-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Optional stability check for top candidates:
```bash
uv run python -m src.train \
  --data-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --run-stability-check \
  --stability-seeds 11,42,87
```

### Make targets
```bash
make train
make evaluate
make serve
make sample-infer
make batch-infer
```

## Deployment Path
FastAPI app:
- `app/main.py`

Endpoints:
- `GET /health`
- `POST /predict/single`
- `POST /predict/batch`

Run API:
```bash
make serve
```

Sample HTTP tests:
```bash
curl -X POST 'http://127.0.0.1:8000/predict/single' \
  -H 'Content-Type: application/json' \
  -d @samples/single_payload.json

curl -X POST 'http://127.0.0.1:8000/predict/batch' \
  -H 'Content-Type: application/json' \
  -d @samples/batch_api_payload.json
```

## Monitoring / Drift / Retraining
Monitoring outputs:
- `artifacts/monitoring/batch_scoring_report.md`
- `artifacts/monitoring/drift_indicators.csv`
- `artifacts/monitoring/recent_batch_scoring.csv`
- `artifacts/monitoring/feature_distribution_comparison.png`

Retraining policy:
1. retrain when >25% monitored features have drift score > 0.20
2. retrain when expected business cost per 1000 worsens by >=20% versus holdout baseline
3. retrain on 30-day cadence when enough fresh labeled data exists

## Key Files
- `end_to_end_ml_training.ipynb`
- `src/train.py`
- `src/data_pipeline.py`
- `src/infer.py`
- `src/evaluate.py`
- `app/main.py`
- `Makefile`
