# Code map

| File | Purpose |
|---|---|
| train.py | Joint data0+data1 XGBoost training and optional separate frozen-model test reporting |
| predict.py | Peer-evaluated predictor; reads JSON/JSON.GZ and writes the required three-column CSV |
| xgb_features.py | Shared 110-feature implementation; unchanged from the trained XGBoost |
| prepare_split.py | Reproduce label-free gene assignments directly from local training metadata |
| evaluate_test.py | Evaluate packaged or explicitly selected model on original test genes |
| validate_submission.py | Check CSV schema, duplicates, finite score range and exact input-site coverage |
| run_pipeline.py | Predict and validate data0/data1/data2 with one explicitly selected model |
| prepare_m6anet.py | Prepare original data0/data1 test sites for official m6Anet inference |
| benchmark_m6anet.py | Align official site probabilities and XGBoost test scores; compute common-site metrics |
| train_baseline.py | Fixed nine-mean logistic baseline; same joint split and separate evaluation |
| evaluate_data2.py | Separate synthetic-data evaluation; binary-label convention must be explicit |
| prepare_m6anet_data2.py | Prepare synthetic data2 for official m6Anet inference |
| test_pipeline.py | Small reproducibility and input/output contract tests |

The required two main scripts are train.py and predict.py. Additional files are support code for reproducibility and benchmarking, not additional models competing with the main submission.

artifacts/selected_xgb.joblib is the NEW joint XGBoost. artifacts/baseline.joblib is labelled as an auxiliary baseline. No old data0-only XGBoost model is included in this new package.

splits/gene_split.csv has gene assignments only, no site labels. example/test.json.gz has unlabelled inputs; expected_predictions.csv contains reference model scores. evaluation/ contains compact reference summary metrics, not label-bearing site tables or raw m6Anet read outputs.
