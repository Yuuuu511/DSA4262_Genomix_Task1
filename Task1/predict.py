"""Predict all sites in new JSON/JSON.GZ without labels or a split manifest."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
from xgb_features import FEATURE_NAMES, VERSION, iter_sites, site_features


def load_model(path):
    bundle = joblib.load(path)

    # Check that the feature names, order and version match the saved model.
    if bundle['feature_names'] != FEATURE_NAMES or bundle['feature_version'] != VERSION:
        raise ValueError('Model feature schema mismatch')

    # The file hash also detects changes to the feature implementation.
    digest = hashlib.sha256(Path(__file__).with_name('xgb_features.py').read_bytes()).hexdigest()
    if bundle.get('feature_hash') != digest:
        raise ValueError('Feature implementation differs from training')

    weight = bundle['positive_weight']
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('Invalid saved positive-class weight')
    return bundle


def predict_file(data, model_path, output, batch_size=2048, overwrite=False):
    data, output = Path(data).resolve(), Path(output).resolve()
    if batch_size < 1:
        raise ValueError('batch_size must be positive')

    # Protect the input data, model and any existing prediction file.
    if output == data or output == Path(model_path).resolve():
        raise ValueError('Output cannot replace input data or model')
    if output.exists() and not overwrite:
        raise FileExistsError('Output exists; choose another path or pass --overwrite')

    bundle = load_model(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    seen = set()
    count = 0
    try:
        # Write to a temporary file so failed predictions do not leave a partial final CSV.
        with tempfile.NamedTemporaryFile('w', newline='', encoding='utf-8',
                                         dir=output.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(['transcript_id', 'transcript_position', 'score'])
            keys, features = [], []

            def flush():
                if not keys:
                    return
                X = np.asarray(features, dtype=np.float32)
                p = bundle['model'].predict_proba(X)[:, 1].astype(float)

                # Adjust for the positive-class weight used during training.
                # When the weight is 1, the scores stay unchanged.
                w = bundle['positive_weight']
                p = p / (w - (w - 1) * p)
                if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError('Invalid predicted scores')
                writer.writerows((tx, pos, float(score)) for (tx, pos), score in zip(keys, p))
                keys.clear()
                features.clear()

            for tx, pos, sequence, reads in iter_sites(data):
                # Detect duplicate sites across the whole input, including earlier batches.
                key = (tx, pos)
                if key in seen:
                    raise ValueError(f'Duplicate site: {key}')
                seen.add(key)
                keys.append(key)
                features.append(site_features(sequence, reads))
                count += 1

                # Predict one batch at a time to limit feature-array memory.
                if len(keys) == batch_size:
                    flush()

            if count == 0:
                raise ValueError('Input contains no sites')

            # Write any remaining sites in the final, smaller batch.
            flush()

        if output.exists() and not overwrite:
            raise FileExistsError('Output appeared during prediction')

        # Move the completed temporary file to the requested output path.
        os.replace(temp_path, output)
        temp_path = None
    finally:
        # Remove the temporary file if prediction failed before completion.
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return count


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=root / 'artifacts/selected_xgb.joblib')
    parser.add_argument('--output', type=Path, default=None, required=True)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    n = predict_file(args.data, args.model, args.output, args.batch_size, args.overwrite)
    print(f'Saved {n:,} predictions to {args.output.resolve()}')


if __name__ == '__main__':
    main()