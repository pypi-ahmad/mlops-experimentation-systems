# Experimentation + Uplift Modeling

Decision-focused experimentation project that combines classical A/B testing with uplift modeling and a four-track modeling workflow in one notebook.

## Project Goal
- Estimate treatment impact (`group B` vs `group A`) on conversion.
- Build robust heterogeneous treatment-effect ranking.
- Compare `ship_all` vs targeted rollout value.
- Select a final model using a unified metric + operations-aware leaderboard.

## Setup and Dataset

```bash
git clone https://github.com/pypi-ahmad/experimentation-uplift-modeling.git
cd experimentation-uplift-modeling
```

- Source used in current run: Kaggle `storytellerman/pharma-ab-test-packaging-impact-in-mobile-app`
- Local files:
  - `data/raw/pharma/pharma_ab_test_data.csv`
  - `experimentation_uplift_modeling.ipynb`
  - `src/ab_test.py`
  - `src/uplift.py`
- Environment: `uv` + Python `3.12.10`
- Note: PyCaret track uses the `4.0.0a8` pre-release API to support Python 3.12.

## Notebook Workflow (15 Sections)
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

## Four Modeling Tracks

### 1) LazyPredict Discovery Lab
- Runs after feature engineering and split strategy setup.
- Benchmarks model families on treatment/control response tasks.
- Applies eligibility filtering and ranking.
- Produces top-3 model families only.

### 2) Manual Engineering Lab
- Implements only the top-3 families selected by LazyPredict.
- Uses manual preprocessing + calibrated T-learners.
- Tunes rollout target fraction on validation split.
- Evaluates uplift metrics + calibration + operational behavior.

### 3) FLAML Optimization Lab
- Trains treatment/control response models with explicit `time_budget`.
- Uses CV-based search and reports:
  - best estimator per arm
  - best config per arm
  - best loss per arm
- Evaluates whether optimization beats manual track on holdout metrics.

### 4) PyCaret Experiment Lab
- Uses PyCaret as a full orchestration track:
  - fit/setup
  - compare_models
  - tune_model
  - calibrate_model
  - finalize_model
  - save_model
- Saves treatment/control artifacts under `artifacts/models/`.

## LazyPredict -> Top 3 Manual Rule
Manual models are not chosen arbitrarily. The notebook enforces:
- LazyPredict discovery ranking
- eligibility filtering
- top-3 only for manual engineering

## Unified Leaderboard Logic
Saved file:
- `artifacts/leaderboard_uplift.csv`

Compared candidates include:
- Baseline model
- LazyPredict top-family prototypes
- Manual top-3 implementations
- Best FLAML candidate
- Best PyCaret finalized candidate

Columns:
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

Ranking uses weighted performance + calibration + efficiency.

## Results Summary
- A/B absolute lift: `+9.31 pp`
- p-value: `0.000273`
- 95% CI: `[+4.31 pp, +14.31 pp]`
- Decision: `ship_all`
- Winner: `Baseline Logistic T-Learner` (current data/run)
- Saved top-3 stability check: `artifacts/top3_seed_stability.csv`

## Deployment and Monitoring Path
- Winner artifact saved to `artifacts/models/winner_t_learner.pkl`.
- PyCaret finalized artifacts saved for treatment/control.
- Monitoring plan in notebook includes:
  - uplift ranking drift (AUUC/Qini)
  - calibration drift
  - segment mix drift
  - retraining triggers and cadence

## Exact Run Instructions

```bash
cd experimentation-uplift-modeling
uv sync
uv run python -m ipykernel install --user --name experimentation-uplift-modeling --display-name "Python (experimentation-uplift-modeling)"
uv run jupyter notebook
```

Execute notebook non-interactively:

```bash
cd experimentation-uplift-modeling
uv run jupyter nbconvert --to notebook --execute experimentation_uplift_modeling.ipynb --output experimentation_uplift_modeling.ipynb
```

Main artifacts:
- `artifacts/leaderboard_uplift.csv`
- `artifacts/top3_seed_stability.csv`
- `artifacts/models/`
