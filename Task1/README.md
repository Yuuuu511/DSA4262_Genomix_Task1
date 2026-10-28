# Task 1: m6A Prediction with XGBoost

**Team:** Genomix  

This project predicts site-level m6A modification scores from m6Anet-processed direct RNA sequencing signals.

- **Training:** `train.py` trains a joint data0+data1 XGBoost classifier.
- **Prediction:** `predict.py` applies a saved model without requiring labels.
- **Pretrained model:** `artifacts/selected_xgb.joblib`.
- **Output:** a CSV containing exactly `transcript_id,transcript_position,score`.

The repository includes a small unlabelled example and a pretrained model. You can install the software and run prediction without downloading the development datasets, retraining the model or installing m6Anet.

## Contents

- [1. Installation](#1-installation)
- [2. Run the example](#2-run-the-example)
- [3. Input and output](#3-input-and-output)
- [4. Predict new datasets](#4-predict-new-datasets)
- [5. Model training](#5-model-training)
- [6. Held-out evaluation and baseline](#6-held-out-evaluation-and-baseline)
- [7. Benchmark against pretrained m6Anet](#7-benchmark-against-pretrained-m6anet)
- [8. Prediction exports and data2 evaluation](#8-prediction-exports-and-data2-evaluation)
- [9. Resources and reproducibility](#9-resources-and-reproducibility)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Repository structure](#11-repository-structure)
- [12. License](#12-license)
- [13. Citation](#13-citation)

## 1. Installation

### 1.1 System requirements

The target environment is:

- Ubuntu 24.04 on AWS/RONIN
- Linux x86_64
- Python 3.11
- CPU only; no GPU required
- Git and curl
- Internet access for installation and repository download

If using AWS/RONIN, connect to the Ubuntu machine using its SSH instructions. Run the installation commands in that terminal.

Check the operating system and architecture:

```bash
uname -s
uname -m
```

Expected output:

```text
Linux
x86_64
```

Check the download tools:

```bash
git --version
curl --version
```

If either tool is missing on a managed machine, ask the administrator to provide it. The following environment installation uses your own account and does not require administrator rights.

### 1.2 Install conda if needed

Check whether conda is available:

```bash
conda --version
```

If conda is already installed, skip the installer.

If Miniforge already exists at `$HOME/miniforge3`, load it:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda --version
```

On a fresh **Linux x86_64** machine without conda, install Miniforge:

```bash
curl -fsSL -o /tmp/Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"

source "$HOME/miniforge3/etc/profile.d/conda.sh"

conda --version
```

Do not rerun the installer over an existing installation.

### 1.3 Create the Python environment

```bash
conda create -n m6a_task1 python=3.11 -y
conda activate m6a_task1

python --version
python -m pip --version
```

Expected: Python **3.11.x**, with Python and pip referring to the `m6a_task1` environment.

If the environment already exists, activate it without recreating it:

```bash
conda activate m6a_task1
```

### 1.4 Download the repository and install dependencies

```bash
git clone --branch main https://github.com/Yuuuu511/DSA4262_Genomix_Task1.git
cd DSA4262_Genomix_Task1/Task1

python -m pip install -r requirements.txt
python -m pip check
```

Expected dependency-check result:

```text
No broken requirements found.
```

If the repository is already downloaded, enter its `Task1` directory instead of cloning again.

The direct dependencies are pinned in [requirements.txt](requirements.txt):

| Package | Version |
|---|---|
| NumPy | 2.2.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| XGBoost | 2.1.4 |
| joblib | 1.4.2 |
| SciPy | 1.15.2 |

**Run the remaining commands from `DSA4262_Genomix_Task1/Task1/`, with `m6a_task1` activated.**

## 2. Run the example

The repository includes:

- `example/test.json.gz`: an unlabelled input containing 20 sites, approximately 60 KiB.
- `example/expected_predictions.csv`: reference model scores, not ground-truth labels.
- `artifacts/selected_xgb.joblib`: the pretrained joint XGBoost model.

No external datasets or labels are required for this example.

### 2.1 Generate predictions

```bash
python predict.py \
  --data example/test.json.gz \
  --output example/predictions.csv
```

Expected terminal message:

```text
Saved 20 predictions to <your Task1 directory>/example/predictions.csv
```

The absolute directory depends on where the repository was downloaded.

### 2.2 Validate the output

```bash
python validate_submission.py \
  --data example/test.json.gz \
  --predictions example/predictions.csv
```

Expected result:

```json
{
  "status": "PASS",
  "rows": 20,
  "score_min": 0.0072179567068815,
  "score_max": 0.8608824014663696
}
```

The validator checks column order, finite scores in `[0,1]`, integer positions, missing values, duplicate sites and exact input-site coverage.

Inspect the CSV:

```bash
head -n 5 example/predictions.csv
wc -l example/predictions.csv
```

Expected beginning:

```csv
transcript_id,transcript_position,score
ENST00000007516,186,0.0784527063369751
ENST00000007516,204,0.1377965360879898
ENST00000007516,307,0.049783166497945786
ENST00000007516,329,0.08629769831895828
```

Expected line count: **21**, consisting of one header and 20 prediction rows.

### 2.3 Run the automated checks

```bash
python test_pipeline.py
```

Expected ending:

```text
Ran 4 tests ...

OK
```

The tests check model and feature integrity, agreement with reference scores within absolute tolerance `1e-6`, batch-size invariance, position zero, duplicate inputs, overwrite protection and malformed reads.

They do not train a new model.

### 2.4 Repeat prediction

Existing prediction files are protected. To intentionally replace the example output:

```bash
python predict.py \
  --data example/test.json.gz \
  --output example/predictions.csv \
  --overwrite
```

Then rerun the validator.

The example verifies installation, prediction execution and output reproducibility. It does not measure biological prediction accuracy because no ground-truth labels are included.

## 3. Input and output

### 3.1 Input format

The predictor accepts plain `.json` and gzip-compressed `.json.gz`.

Input must contain newline-delimited JSON in the course's m6Anet-processed signal format. Each line is a complete JSON object nested as:

```text
transcript_id -> transcript_position -> seven-base sequence -> reads
```

Each site must contain one seven-base sequence and at least one read.

Each read contains nine finite numeric features: dwell time, signal standard deviation and mean current for the minus-one, central and plus-one signal segments, in that order.

Example schema:

```json
{"tx_example":{"0":{"AAGACCA":[[0.01,2,100,0.02,3,110,0.03,4,90]]}}}
```

Dwell times must be nonnegative. Sequence bases must be A/C/G/T; U is converted to T.

FASTQ files, raw nanopore signals, label tables and m6Anet probability CSVs cannot be used directly as prediction input.

### 3.2 Output format

The output contains exactly these columns:

```csv
transcript_id,transcript_position,score
```

| Column | Meaning |
|---|---|
| `transcript_id` | Original transcript identifier |
| `transcript_position` | Original integer transcript position, including zero |
| `score` | Continuous m6A prediction score in `[0,1]` |

The predictor writes one row per valid input site. It preserves transcript identifiers and positions without converting them to genomic coordinates.

Higher scores indicate stronger model support for m6A modification. Scores are not thresholded into binary calls.

The scores have not been empirically calibrated as modification fractions. A score of 0.8 should not be interpreted as 80% of molecules being modified.

## 4. Predict new datasets

Replace the input path with an existing processed file and choose a new output filename:

```bash
python predict.py \
  --data /path/to/new_dataset.json.gz \
  --output my_predictions.csv

python validate_submission.py \
  --data /path/to/new_dataset.json.gz \
  --predictions my_predictions.csv
```

Prediction requires no labels, gene IDs, split manifest or retraining.

The default model is `artifacts/selected_xgb.joblib`. To select a separately retrained model:

```bash
python predict.py \
  --data /path/to/new_dataset.json.gz \
  --model artifacts/retrained/selected_xgb.joblib \
  --output retrained_predictions.csv
```

All valid sites are scored, including sites with fewer than 20 reads. Duplicate sites and invalid reads stop prediction explicitly.

`--data` specifies the input and `--output` specifies the destination CSV. Use `--model` to select another compatible model, `--overwrite` to replace an existing prediction file, and `--batch-size` to change the number of sites processed per batch. The default batch size is 2,048.

Explicit relative paths are resolved against the current terminal directory. The default model path is resolved relative to `predict.py`.

View command usage:

```bash
python predict.py --help
python validate_submission.py --help
```

## 5. Model training

### 5.1 Obtain and arrange the development data

Download the datasets and labels from the **DSA4262 Team Project materials on Canvas**. These files are not included in this repository and are not needed for the prediction example.

Place the dataset folders alongside `Task1/`:

```text
DSA4262_Genomix_Task1/
├── data0/
│   ├── dataset0.json.gz
│   └── data.info.labelled
├── data1/
│   ├── dataset1.json.gz
│   └── data.info
├── data2/
│   ├── dataset2.json.gz
│   └── data.info
└── Task1/
    └── ...
```

Training uses **data0 and data1 only**. Data2 is used for evaluation in Section 8.

From `Task1/`, set the data directory:

```bash
DATA_ROOT="$(cd .. && pwd)"
```

If the datasets are stored elsewhere, use their parent directory instead:

```bash
DATA_ROOT="/absolute/path/to/your/data_directory"
```

### 5.2 Check metadata and split membership

```bash
python train.py --check-only \
  --labels0 "$DATA_ROOT/data0/data.info.labelled" \
  --labels1 "$DATA_ROOT/data1/data.info" \
  --manifest splits/gene_split.csv
```

This validates metadata and split membership without training. It does not perform a full raw-JSON coverage check.

### 5.3 Train the joint model

```bash
python -u train.py \
  --data0 "$DATA_ROOT/data0/dataset0.json.gz" \
  --data1 "$DATA_ROOT/data1/dataset1.json.gz" \
  --labels0 "$DATA_ROOT/data0/data.info.labelled" \
  --labels1 "$DATA_ROOT/data1/data.info" \
  --manifest splits/gene_split.csv \
  --output-dir artifacts/retrained
```

The main output is:

```text
artifacts/retrained/selected_xgb.joblib
```

Training also writes validation metrics, a training report and local audit tables.

The packaged model is not replaced. A nonempty training output directory is refused; use a new directory for another run. Automatic resume is not implemented.

### 5.4 Features and model selection

The model uses 110 site-level features:

| Feature block | Number |
|---|---:|
| Five summary statistics for nine signals | 45 |
| Log-dwell summaries | 6 |
| Central/flanking signal contrasts | 12 |
| Log read count | 1 |
| Seven-base sequence encoding | 28 |
| Central motif encoding | 18 |
| Total | 110 |

Gene IDs, transcript IDs, positions, source IDs and labels are not classifier features.

Four configurations vary maximum depth `{3,5}` and positive-class weight `{1,4}`. Early stopping uses validation `aucpr`. Configuration selection maximizes the average of validation ROC-AUC and trapezoidal PR-AUC.

The selected configuration and round count are refitted on training plus validation observations only. The packaged model has depth 3, positive weight 1 and 912 boosting rounds.

### 5.5 Gene-disjoint split

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| data0 | 73,233 | 24,773 | 23,832 |
| data1 | 45,146 | 15,340 | 15,187 |
| Combined | 118,379 | 40,113 | 39,019 |

The final fit uses 158,492 training-plus-validation observations.

An additional 15,137 data1 sites without reliable gene mapping are excluded from training/evaluation membership. Inference can still score them.

The same gene belongs to one partition across both datasets. Genes, transcripts and shared sites cannot cross partitions. Data2 and the new final-test datasets are not used for training or model selection.

The public split file contains only `gene_id,split`, without ground-truth site labels.

To reproduce the assignments:

```bash
python prepare_split.py \
  --data0-labels "$DATA_ROOT/data0/data.info.labelled" \
  --data1-labels "$DATA_ROOT/data1/data.info" \
  --output splits/reproduced_gene_split.csv \
  --check-against splits/gene_split.csv
```

The split uses 20% of genes for test with seed 42, then 25% of the remaining genes for validation with seed 43.

For different training data, generate a suitable new split and pass it through `--manifest`. Do not choose split seeds based on test performance.

## 6. Held-out evaluation and baseline

These commands require the development data and `DATA_ROOT` configured in Section 5.

### 6.1 Evaluate the packaged XGBoost model

```bash
python evaluate_test.py \
  --data0 "$DATA_ROOT/data0/dataset0.json.gz" \
  --data1 "$DATA_ROOT/data1/dataset1.json.gz" \
  --labels0 "$DATA_ROOT/data0/data.info.labelled" \
  --labels1 "$DATA_ROOT/data1/data.info" \
  --evaluation-dir evaluation/recomputed
```

This evaluates the saved model without fitting.

Outputs include:

```text
evaluation/recomputed/test_metrics.csv
evaluation/recomputed/heldout_test_predictions.csv
evaluation/recomputed/evaluation_report.json
```

To evaluate a newly trained model, add `--model artifacts/retrained/selected_xgb.joblib` and choose a new evaluation directory.

Evaluation requires labels; prediction does not.

### 6.2 Train and evaluate the baseline

The fixed baseline uses nine signal means, a training-fitted `StandardScaler` and balanced logistic regression on the same partitions.

Train:

```bash
python train_baseline.py \
  --data0 "$DATA_ROOT/data0/dataset0.json.gz" \
  --data1 "$DATA_ROOT/data1/dataset1.json.gz" \
  --labels0 "$DATA_ROOT/data0/data.info.labelled" \
  --labels1 "$DATA_ROOT/data1/data.info" \
  --output-dir artifacts/retrained_baseline
```

Evaluate:

```bash
python train_baseline.py --evaluate-test \
  --data0 "$DATA_ROOT/data0/dataset0.json.gz" \
  --data1 "$DATA_ROOT/data1/dataset1.json.gz" \
  --labels0 "$DATA_ROOT/data0/data.info.labelled" \
  --labels1 "$DATA_ROOT/data1/data.info" \
  --output-dir artifacts/retrained_baseline
```

The reference baseline and metrics are included as:

```text
artifacts/baseline.joblib
evaluation/baseline_test_metrics.csv
```

### 6.3 Reference results

| Model | Overall ROC-AUC | Overall trapezoidal PR-AUC |
|---|---:|---:|
| Joint XGBoost | 0.881557 | 0.464971 |
| Pretrained m6Anet | 0.878555 | 0.446030 |
| Nine-mean logistic baseline | 0.688417 | 0.112339 |

These results cover 39,019 held-out source/site observations. Average precision is reported separately from trapezoidal PR-AUC.

See [RESULTS.md](RESULTS.md) for dataset-specific metrics and methodological limitations.

The original test partitions were inspected during earlier development. These metrics are exploratory reporting rather than a newly untouched final test. Do not continue tuning based on them.

## 7. Benchmark against pretrained m6Anet

The benchmark prepares the same held-out sites for official pretrained m6Anet inference and compares both methods on common sites.

m6Anet is not required for XGBoost prediction.

### 7.1 Prepare held-out observations

In the `m6a_task1` environment:

```bash
python prepare_m6anet.py \
  --data0 "$DATA_ROOT/data0/dataset0.json.gz" \
  --data1 "$DATA_ROOT/data1/dataset1.json.gz" \
  --labels0 "$DATA_ROOT/data0/data.info.labelled" \
  --labels1 "$DATA_ROOT/data1/data.info" \
  --output-dir evaluation/m6anet_inputs
```

The adapter preserves the nine signal values, appends synthetic local read IDs, rebuilds byte offsets and excludes sites with fewer than 20 reads.

### 7.2 Create the m6Anet environment

Use a separate Python 3.8 conda environment on Linux x86_64:

```bash
conda create -n m6anet38 --override-channels \
  -c pytorch -c conda-forge -c bioconda -c defaults \
  python=3.8 m6anet=2.0.2 cpuonly 'numpy<1.24' -y

conda activate m6anet38
```

Skip creation if the environment already exists. Keep it separate from the XGBoost environment.

### 7.3 Run official inference

From the same `Task1` directory:

```bash
for source in data0 data1
do
  m6anet inference \
    --input_dir "evaluation/m6anet_inputs/$source" \
    --out_dir "evaluation/m6anet_outputs/$source" \
    --n_processes 1 \
    --num_iterations 1000 \
    --seed 42 \
    --save_per_batch 100000
done
```

The large `save_per_batch` value is a workaround for an observed final-batch saving issue in m6Anet 2.0.2. It must exceed the number of inference batches.

### 7.4 Score common sites

Generate the held-out XGBoost predictions in Section 6 first, then run:

```bash
conda activate m6a_task1

python benchmark_m6anet.py \
  --xgb-predictions evaluation/recomputed/heldout_test_predictions.csv \
  --labels0 "$DATA_ROOT/data0/data.info.labelled" \
  --labels1 "$DATA_ROOT/data1/data.info" \
  --m6anet-data0 evaluation/m6anet_outputs/data0/data.site_proba.csv \
  --m6anet-data1 evaluation/m6anet_outputs/data1/data.site_proba.csv \
  --output-dir evaluation/recomputed_m6anet
```

Comparison uses site-level `probability_modified`, not `mod_ratio` or individual-read probabilities.

Missing scores are excluded from both methods' common-site comparison, and coverage is reported. The reference comparison covers 39,019/39,019 sites under `evaluation/m6anet_comparison/`.

XGBoost and m6Anet differ in features, training data, normalization and aggregation. Pretrained m6Anet overlap with course data is unknown, so this comparison does not establish superiority under matched independent training conditions.

## 8. Prediction exports and data2 evaluation

### 8.1 Generate predictions for the original datasets

With all three datasets under `DATA_ROOT`:

```bash
python run_pipeline.py \
  --data-root "$DATA_ROOT" \
  --output-dir submissions \
  --team-name Genomix_haribo
```

Expected files:

| File | Prediction rows |
|---|---:|
| `Genomix_haribo_dataset0.csv` | 121,838 |
| `Genomix_haribo_dataset1.csv` | 90,810 |
| `Genomix_haribo_dataset2.csv` | 1,323 |

The wrapper predicts and validates each dataset using the same saved model. Existing prediction files are refused.

Use the registered team name and current course filename instructions for submission.

### 8.2 Evaluate fractional data2 labels

Data2 is not used for training or tuning. Labels >= 0.5 are positive and labels < 0.5 are negative.

For `label >= 0.5`:

```bash
python evaluate_data2.py \
  --labels "$DATA_ROOT/data2/data.info" \
  --predictions submissions/Genomix_haribo_dataset2.csv \
  --label-rule ge0.5 \
  --output-dir evaluation/recomputed_data2_ge0_5
```

The script uses the confirmed ge0.5 label rule by default. Prediction scores remain continuous; the threshold is applied only to the ground-truth labels.

Reference results are summarized in [RESULTS.md](RESULTS.md).

### 8.3 Data2 m6Anet comparison

Prepare all eligible data2 sites in the XGBoost environment:

```bash
conda activate m6a_task1

python prepare_m6anet_data2.py \
  --data "$DATA_ROOT/data2/dataset2.json.gz" \
  --output-dir evaluation/m6anet_inputs/data2
```

Run official inference in the m6Anet environment:

```bash
conda activate m6anet38

m6anet inference \
  --input_dir evaluation/m6anet_inputs/data2 \
  --out_dir evaluation/m6anet_outputs/data2 \
  --n_processes 1 \
  --num_iterations 1000 \
  --seed 42 \
  --save_per_batch 100000
```

Compare XGBoost, the saved logistic baseline and pretrained m6Anet on the same data2 sites:

```bash
conda activate m6a_task1

python evaluate_data2.py \
  --data "$DATA_ROOT/data2/dataset2.json.gz" \
  --labels "$DATA_ROOT/data2/data.info" \
  --predictions submissions/Genomix_haribo_dataset2.csv \
  --baseline-model artifacts/baseline.joblib \
  --m6anet-predictions evaluation/m6anet_outputs/data2/data.site_proba.csv \
  --output-dir evaluation/recomputed_data2_comparison
```

The script uses the confirmed label rule: labels >= 0.5 are positive and labels < 0.5 are negative. Prediction scores remain continuous.

This comparison is separate from the data0/data1 held-out benchmark. Missing m6Anet sites are reported, and all three models are compared on identical sites in the `data2_common_m6anet` results. The baseline is evaluated without retraining.

### 8.4 New unlabelled final datasets

Use Section 4's prediction and validation commands separately for each new input, with a unique output filename.

`run_pipeline.py` only targets the original datasets 0–2. The new datasets are not used for training, preprocessing fits, split selection or hyperparameter selection.

## 9. Resources and reproducibility

Prediction streams the input and processes up to 2,048 feature vectors per batch by default.

It also retains site identifiers to detect duplicates and parses each JSON line in memory. Total memory depends on site count and the size of the largest line.

To reduce feature-batch memory:

```bash
python predict.py \
  --data example/test.json.gz \
  --output example/predictions_small_batch.csv \
  --batch-size 256
```

Training requires more time and memory than the small example because it extracts training/validation features and fits four candidates plus a final model. m6Anet benchmarking also creates intermediate files on disk.

### Verification status

The prediction workflow was tested on an AWS/RONIN Ubuntu machine:

- Dependency installation succeeded.
- All four automated tests passed.
- Example prediction produced 20 rows.
- Output validation returned `PASS`.

Full model training and m6Anet inference were not included in this check.

## 10. Troubleshooting

| Problem | Suggested action |
|---|---|
| `conda: command not found` | Install conda or source the shell script for an existing installation. |
| Python is not 3.11 | Activate `m6a_task1` and check `python --version`. |
| Missing Python package | Activate the environment and run `python -m pip install -r requirements.txt`. |
| GitHub requests authentication | Check the repository URL and access permissions. |
| Script or input file not found | Check the working directory and supplied path. |
| `libgomp.so.1` missing on Ubuntu | Run `conda install -c conda-forge libgomp -y` in the active environment, then retry. |
| Prediction output already exists | Choose a new filename or use `--overwrite` for intentional replacement. |
| Training/evaluation directory is nonempty | Choose a new output directory. |
| Feature implementation or schema mismatch | Restore the matching model and `xgb_features.py` from the same version. |
| Invalid reads or sequence | Check the processed input schema in Section 3. |
| Duplicate site or coverage mismatch | Check transcript-position uniqueness and use the same input for prediction and validation. |
| Reference scores differ | Check dependency versions, model and commit; do not edit reference scores to hide differences. |

Each executable script provides usage information through `--help`, for example:

```bash
python predict.py --help
python train.py --help
python evaluate_test.py --help
python evaluate_data2.py --help
```

## 11. Repository structure

| File or directory | Purpose |
|---|---|
| `train.py` / `predict.py` | Training and prediction entry points |
| `xgb_features.py` | Shared feature implementation |
| `test_pipeline.py` / `validate_submission.py` | Prediction checks |
| `prepare_split.py` / `splits/` | Gene split generation and assignments |
| `evaluate_test.py` | Frozen-model held-out evaluation |
| `train_baseline.py` | Fixed logistic baseline |
| `prepare_m6anet.py` / `benchmark_m6anet.py` | Held-out m6Anet preparation and comparison |
| `prepare_m6anet_data2.py` / `evaluate_data2.py` | Data2 preparation and evaluation |
| `run_pipeline.py` | Prediction exports for datasets 0–2 |
| `example/` | Small unlabelled input and reference scores |
| `artifacts/` | Saved models and training summaries |
| `evaluation/` | Reference evaluation summaries |
| [RESULTS.md](RESULTS.md) | Measured results and limitations |
| [CODE_MAP.md](CODE_MAP.md) | Supporting script descriptions |

## 12. License

This project is licensed under the MIT License. See [LICENSE](../LICENSE.md) for details.

Third-party software and datasets retain their own terms.

## 13. Citation

When referring to this project, cite:

> Genomix team (2026). *DSA4262_Genomix_Task1: m6A prediction from direct RNA sequencing*. [GitHub repository](https://github.com/Yuuuu511/DSA4262_Genomix_Task1).

Include the exact commit hash used. No DOI or formal release identifier has been assigned.

Cite third-party methods using their official guidance:

- [XGBoost](https://github.com/dmlc/xgboost)
- [scikit-learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn)
- [m6Anet](https://github.com/GoekeLab/m6anet)
