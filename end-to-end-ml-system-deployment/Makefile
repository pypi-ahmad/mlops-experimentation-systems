DATA_PATH ?= data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
MODEL_PATH ?= artifacts/model_registry/final_model.joblib

.PHONY: download-data train evaluate serve sample-infer batch-infer

download-data:
	uv run --with kaggle kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip

train:
	uv run python -m src.train --data-path $(DATA_PATH)

evaluate:
	uv run python -m src.evaluate --train-data-path $(DATA_PATH) --model-path $(MODEL_PATH)

serve:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

sample-infer:
	uv run python -m src.infer --input-file samples/single_payload.json --model-path $(MODEL_PATH)

batch-infer:
	uv run python -m src.infer --input-file samples/batch_payload.json --model-path $(MODEL_PATH)
