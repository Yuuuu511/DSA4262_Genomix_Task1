"""Check prediction schema, probability range and exact input site coverage."""
import argparse
import json
import numpy as np
import pandas as pd
from xgb_features import iter_sites


def validate(data, predictions):
    # Keep transcript IDs as strings when reading the CSV.
    p = pd.read_csv(predictions, dtype={'transcript_id': str})

    # The submission requires these three columns in this order.
    if p.columns.tolist() != ['transcript_id', 'transcript_position', 'score']:
        raise ValueError('Expected transcript_id,transcript_position,score in this order')
    if p.empty or p.isna().any().any() or p.duplicated(['transcript_id', 'transcript_position']).any():
        raise ValueError('Empty predictions, missing values or duplicate sites')

    scores = pd.to_numeric(p.score, errors='raise')
    positions = pd.to_numeric(p.transcript_position, errors='raise')

    # Reject infinite values and scores outside the probability range.
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError('Scores must be finite probabilities')
    if not np.isfinite(positions).all() or (positions % 1 != 0).any():
        raise ValueError('Positions must be finite integers')

    # Collect the unique transcript-position pairs from the input.
    expected = set()
    for tx, pos, _, _ in iter_sites(data):
        if (tx, pos) in expected:
            raise ValueError('Duplicate raw input site')
        expected.add((tx, pos))

    # Compare site sets so CSV row order does not affect validation.
    actual = set(zip(p.transcript_id, positions.astype(np.int64)))
    if actual != expected:
        raise ValueError(f'Coverage mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')

    return {'status': 'PASS', 'rows': len(p), 'score_min': float(scores.min()),
            'score_max': float(scores.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()

    # Print a short validation summary in JSON format.
    print(json.dumps(validate(args.data, args.predictions), indent=2))


if __name__ == '__main__':
    main()