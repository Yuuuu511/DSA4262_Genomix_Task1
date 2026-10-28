"""Evaluate data2 using the label threshold of 0.5.

Evaluate existing XGBoost predictions, generate predictions with a saved
logistic baseline, and compare with existing m6Anet site probabilities.
No model or scaler is fitted during evaluation.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

KEYS = ["transcript_id", "transcript_position"]


def load_table(path, value_column, label=False):
    """Read and validate one value per transcript-position site."""
    frame = pd.read_csv(path, dtype={"transcript_id": str})

    required = KEYS + [value_column]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{path}: requires {required}")

    frame = frame[required].copy()

    if (
        frame.isna().any().any()
        or frame.transcript_id.str.strip().eq("").any()
    ):
        raise ValueError(f"{path}: missing values or empty identifiers")

    positions = pd.to_numeric(
        frame.transcript_position, errors="raise"
    )
    if (
        not np.isfinite(positions).all()
        or (positions % 1 != 0).any()
    ):
        raise ValueError(f"{path}: positions must be finite integers")

    frame["transcript_position"] = positions.astype(np.int64)

    # Unique site keys prevent ambiguous matches between prediction files.
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate sites")

    values = pd.to_numeric(frame[value_column], errors="raise")
    if (
        not np.isfinite(values).all()
        or not values.between(0, 1).all()
    ):
        kind = "labels" if label else "probability scores"
        raise ValueError(f"{path}: {kind} must be finite and in [0, 1]")

    frame[value_column] = values

    if frame.empty:
        raise ValueError(f"{path}: empty table")

    return frame


def metrics(name, dataset, y, scores):
    """Calculate ranking metrics from continuous prediction scores."""
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"{dataset}: both binary classes required for ROC/PR evaluation"
        )

    # Trapezoidal PR-AUC and average precision are different summaries.
    precision, recall, _ = precision_recall_curve(y, scores)

    return {
        "model": name,
        "dataset": dataset,
        "n_sites": len(y),
        "positive_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(y, scores)),
    }


def baseline_predictions(data_path, model_path):
    """Apply the saved baseline without refitting its scaler or classifier."""
    from predict import load_model, predict_file

    # Reuse the existing model/feature compatibility checks.
    bundle = load_model(model_path)

    expected_scope = (
        "data0+data1 train+validation; nine-mean logistic baseline"
    )
    if bundle.get("training_data") != expected_scope:
        raise ValueError(
            "Expected the saved nine-mean logistic baseline bundle"
        )

    # The logistic baseline must not receive XGBoost weight correction.
    if bundle.get("positive_weight") != 1:
        raise ValueError(
            "The logistic baseline must have positive_weight=1"
        )

    if not np.array_equal(bundle["model"].classes_, [0, 1]):
        raise ValueError("Expected baseline classes ordered as [0, 1]")

    # The saved pipeline selects the nine signal means and applies its
    # already-fitted scaler. predict_file never calls fit().
    # With positive_weight=1, its score correction leaves scores unchanged.
    with tempfile.TemporaryDirectory(prefix="data2_baseline_") as directory:
        output = Path(directory) / "baseline.csv"
        predict_file(data_path, model_path, output)

        return load_table(output, "score").rename(
            columns={"score": "baseline_score"}
        )


def evaluate(
    labels_path,
    predictions_path,
    output_dir,
    m6anet_path=None,
    label_rule="ge0.5",
    data_path=None,
    baseline_model=None,
):
    # label >= 0.5 is positive; otherwise negative.
    # Reject other rules so comparisons cannot silently use a different target.
    if label_rule != "ge0.5":
        raise ValueError("The confirmed data2 label rule is ge0.5")

    # Both inputs are needed for baseline inference. Omitting both
    # preserves the original XGBoost/m6Anet evaluation behavior.
    if (data_path is None) != (baseline_model is None):
        raise ValueError("Supply --data and --baseline-model together")

    out = Path(output_dir)
    if out.exists() and any(
        p.name != ".DS_Store" for p in out.iterdir()
    ):
        raise FileExistsError(
            "Data2 evaluation output directory is nonempty"
        )

    labels = load_table(
        labels_path, "label", label=True
    ).rename(columns={"label": "label_original"})

    # Keep fractional labels for auditing; include exactly 0.5 as positive.
    # This threshold applies to labels only, never to prediction scores.
    labels["label_binary"] = labels.label_original.ge(0.5).astype(int)

    pred = load_table(predictions_path, "score").rename(
        columns={"score": "xgb_score"}
    )

    # An outer merge detects both missing predictions and extra sites.
    aligned = labels.merge(
        pred,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError(
            "XGBoost predictions must cover exactly all data2 label sites"
        )
    aligned = aligned.drop(columns="_merge")

    models = [("xgboost", "xgb_score")]

    if baseline_model is not None:
        baseline = baseline_predictions(data_path, baseline_model)

        # The baseline must cover the same complete set of labelled sites.
        aligned = aligned.merge(
            baseline,
            on=KEYS,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not aligned["_merge"].eq("both").all():
            raise ValueError(
                "Baseline input must cover exactly all data2 label sites"
            )
        aligned = aligned.drop(columns="_merge")
        models.append(("logistic_9means", "baseline_score"))

    # Report full-data metrics separately from common-m6Anet-site metrics.
    results = [
        metrics(
            name,
            "data2_all",
            aligned.label_binary,
            aligned[column],
        )
        for name, column in models
    ]

    report = {
        "label_rule": label_rule,
        "label_rule_status": (
            "label >= 0.5 is 1; label < 0.5 is 0."
        ),
        "n_sites": len(aligned),
        "n_original_labels_equal_0_5": int(
            labels.label_original.eq(0.5).sum()
        ),
        "positive_sites": int(labels.label_binary.sum()),
        "negative_sites": int(labels.label_binary.eq(0).sum()),
        "baseline_evaluated": baseline_model is not None,
        "notes": [
            "Data2 is not used for training or tuning.",
            "Fractional labels use threshold of 0.5.",
            "Prediction scores remain continuous; no score threshold is applied.",
        ],
    }

    if baseline_model is not None:
        report["baseline_model"] = str(Path(baseline_model).resolve())
        report["baseline_data"] = str(Path(data_path).resolve())
        report["notes"].append(
            "Baseline uses the saved data0+data1 pipeline without refitting."
        )

    missing = None

    if m6anet_path:
        # Use site-level probabilities, not modification ratios.
        m6anet = load_table(
            m6anet_path, "probability_modified"
        ).rename(columns={"probability_modified": "m6anet_score"})

        comparison = aligned.merge(
            m6anet,
            on=KEYS,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        if comparison["_merge"].eq("right_only").any():
            raise ValueError("m6Anet returned sites outside data2 labels")

        # Do not replace missing m6Anet scores with zero.
        missing = comparison.loc[
            comparison["_merge"].eq("left_only"), KEYS
        ].copy()

        common = comparison.loc[
            comparison["_merge"].eq("both")
        ].copy()

        if common.empty:
            raise ValueError(
                "No shared data2 sites for m6Anet comparison"
            )

        # All models are compared on identical sites and binary labels.
        common_models = models + [
            ("m6anet_pretrained", "m6anet_score")
        ]
        for name, column in common_models:
            results.append(
                metrics(
                    name,
                    "data2_common_m6anet",
                    common.label_binary.astype(int),
                    common[column],
                )
            )

        report["m6anet_common_sites"] = len(common)
        report["m6anet_missing_sites"] = len(missing)
        report["notes"].append(
            "Pretrained m6Anet training overlap is unknown; "
            "comparisons use shared sites."
        )

        aligned = aligned.merge(
            m6anet,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

    # Write only after validation and metric calculation succeed.
    out.mkdir(parents=True, exist_ok=True)

    result_table = pd.DataFrame(results)
    result_table.to_csv(out / "data2_metrics.csv", index=False)

    # This audit file contains labels and should remain local.
    aligned.to_csv(
        out / "data2_evaluated_predictions.csv", index=False
    )

    (out / "data2_evaluation_report.json").write_text(
        json.dumps(report, indent=2)
    )

    if missing is not None:
        missing.to_csv(out / "missing_m6anet_sites.csv", index=False)

    print(result_table.to_string(index=False))
    print(
        f"Label rule: {label_rule}. "
        f"Original labels exactly 0.5: "
        f"{report['n_original_labels_equal_0_5']}."
    )
    print(f"Saved {out.resolve()}")

    return results, report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--labels",
        type=Path,
        default=root.parent / "data2/data.info",
        help="Local data2 table containing fractional labels",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root / "submissions/Genomix_haribo_dataset2.csv",
        help="Existing XGBoost data2 prediction CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "evaluation/recomputed_data2",
        help="New or empty evaluation output directory",
    )
    parser.add_argument(
        "--label-rule",
        choices=["ge0.5"],
        default="ge0.5",
        help="Confirmed rule: label >= 0.5 is 1; otherwise 0 (default: ge0.5)",
    )
    parser.add_argument(
        "--m6anet-predictions",
        type=Path,
        help="Existing official data2 data.site_proba.csv",
    )

    # Supply both to include frozen-baseline predictions.
    parser.add_argument(
        "--data",
        type=Path,
        help="Processed data2 JSON/JSON.GZ for baseline inference",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        help="Saved baseline.joblib trained on data0+data1",
    )

    args = parser.parse_args()

    if (args.data is None) != (args.baseline_model is None):
        parser.error("--data and --baseline-model must be supplied together")

    evaluate(
        args.labels,
        args.predictions,
        args.output_dir,
        m6anet_path=args.m6anet_predictions,
        label_rule=args.label_rule,
        data_path=args.data,
        baseline_model=args.baseline_model,
    )


if __name__ == "__main__":
    main()