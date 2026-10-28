# Measured results: bundled joint XGBoost

## Method and features

The main model is the user's selected data0+data1 XGBoost (depth 3, positive weight 1, 912 rounds). Four candidates were trained on 118,379 source/site observations and selected using 40,113 validation observations; final fitting used 158,492 train+validation observations. The same gene belongs to one partition across both sources. Another 15,137 data1 observations were excluded because their gene could not be mapped from the supplied metadata.

Each position is represented by 110 features from its variable number of reads:

| Feature block | Number |
|---|---:|
| Mean, standard deviation, q10, median, q90 for each of nine signals | 45 |
| Mean/standard deviation of log-transformed dwell time at three positions | 6 |
| Mean/standard deviation of centre-minus-left/right signal contrasts | 12 |
| log1p read count | 1 |
| Seven-base sequence one-hot encoding | 28 |
| Central DRACH motif one-hot encoding | 18 |
| Total | 110 |

Gene/transcript identifiers, source identity and positions are not classifier features. Positions/IDs are used to join and output predictions. Signal distributions beyond means and explicit read count address variable reads per site. Positive class weights 1 and 4 were tested; weight 1 was selected. ROC-AUC, trapezoidal PR-AUC and AP are all reported because positives are uncommon. No empirical calibration or modification-fraction estimation was performed.

## Original held-out genes

All methods below are scored on identical 39,019 source/site observations: 23,832 data0 and 15,187 data1. Reference m6Anet coverage is 39,019/39,019. PR-AUC below is trapezoidal area, not AP.

| Model | Dataset | ROC-AUC | PR-AUC | AP |
|---|---|---:|---:|---:|
| Joint XGBoost | overall | 0.881557 | 0.464971 | 0.465312 |
| Pretrained m6Anet | overall | 0.878555 | 0.446030 | 0.446313 |
| Nine-mean logistic baseline | overall | 0.688417 | 0.112339 | 0.112523 |
| Joint XGBoost | data0 | 0.924625 | 0.526550 | 0.527149 |
| Pretrained m6Anet | data0 | 0.935117 | 0.527973 | 0.528455 |
| Nine-mean logistic baseline | data0 | 0.705415 | 0.099830 | 0.100140 |
| Joint XGBoost | data1 | 0.831733 | 0.407489 | 0.408143 |
| Pretrained m6Anet | data1 | 0.813147 | 0.369707 | 0.370311 |
| Nine-mean logistic baseline | data1 | 0.666076 | 0.129998 | 0.130422 |

Overall positive prevalence is 6.0201%; the theoretical no-skill precision reference is 0.060201 and random ROC-AUC reference is 0.5. These are reference values, not measured outcomes of a simulated random classifier.

XGBoost scores numerically higher overall and on data1 than pretrained m6Anet, while m6Anet scores slightly higher on data0. No significance testing was performed. Overall pooling weights every source/site observation once; it does not show superiority in every source or on unseen final datasets.

The logistic baseline uses only nine signal means, a training-fitted StandardScaler and fixed balanced LogisticRegression. It was trained during package preparation, on the exact same train/validation membership. Its final fit includes validation after the fixed method was assessed. It is deliberately simple; its weakness does not by itself establish that every engineered feature is necessary.

## Separate synthetic data2 results

All 1,323 data2 sites were evaluated. No data2 observations were used to train or tune XGBoost or the logistic baseline.

Fractional labels are converted to binary labels using **label >= 0.5 as positive and label < 0.5 as negative**. The resulting positive rate is 71.43%. Prediction scores remain continuous.

All three models are compared on the same 1,323 sites, with complete m6Anet coverage. PR-AUC denotes trapezoidal area; average precision (AP) is reported separately.

| Model | Sites | Positive rate | ROC-AUC | PR-AUC | AP |
|---|---:|---:|---:|---:|---:|
| Joint XGBoost | 1,323 | 71.43% | 0.684656 | 0.857501 | 0.857608 |
| Nine-mean logistic baseline | 1,323 | 71.43% | 0.415946 | 0.669080 | 0.669993 |
| Pretrained m6Anet | 1,323 | 71.43% | 0.673856 | 0.864178 | 0.864272 |

XGBoost outperformed the logistic baseline on all three metrics. Compared with pretrained m6Anet, XGBoost achieved a slightly higher ROC-AUC, while m6Anet achieved slightly higher PR-AUC and AP. No significance testing was performed.

## Benchmark differences and limitations

- XGBoost uses engineered site summaries and learns supervised labels from data0 and data1. m6Anet uses a read-based multiple-instance model and pretrained normalization/weights; the default human model is HCT116-based. See [official m6Anet documentation](https://m6anet.readthedocs.io/en/latest/) and [project repository](https://github.com/GoekeLab/m6anet).
- Official m6Anet inference ran previously on AWS with Python 3.8, PyTorch 1.6.0, m6Anet 2.0.2, seed 42, 1,000 sampling iterations and one process. Scoring uses site-level probability_modified, not mod_ratio or individual-read probabilities. This package only rescored those existing outputs.
- m6Anet applies a minimum-read rule. Every eligible original test site met 20 reads here, so full common-site coverage is possible. Our predictor emits a score for every valid input site even at lower depth; future coverage/protocol differences must be reported.
- Pretrained m6Anet training overlap is unknown. It is not a matched-training experiment, and no paper AUC is directly substituted into this table.
- The original test set has been examined during earlier model development/comparisons. Results are exploratory and should not be described as a newly untouched final holdout. Model and split selection must not continue using these scores.
- Data0/data1 training sources, chemistry and labels need not match four new unlabelled final datasets. These existing results do not guarantee final leaderboard position.
- The old data0-only model was selected on data0 validation, whereas the joint model was selected on combined validation. Their historical comparison changes selection population as well as training-source scope; it is not a strict one-factor ablation.
- All-source leaderboard CSVs include fitted observations and cannot establish independent generalization. Use the held-out table with its stated limitations for reporting.

The existing frozen XGBoost was not retrained during packaging. Label-bearing bundle metadata was removed, while feature code, fitted booster and example predictions were verified unchanged. The prediction verification scope is described in the README.
