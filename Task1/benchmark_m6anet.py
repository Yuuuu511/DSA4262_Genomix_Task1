"""Score XGBoost and existing official m6Anet predictions on identical test sites.

This scores prediction CSVs. It does not install or execute m6Anet inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from train import KEYS, metrics, build_manifest


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xgb-predictions', type=Path, default=root / 'evaluation/recomputed/heldout_test_predictions.csv')
    parser.add_argument('--manifest', type=Path, default=root / 'splits/gene_split.csv')
    parser.add_argument('--labels0', type=Path, default=root.parent / 'data0/data.info.labelled')
    parser.add_argument('--labels1', type=Path, default=root.parent / 'data1/data.info')
    parser.add_argument('--m6anet-data0', type=Path, required=True)
    parser.add_argument('--m6anet-data1', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=root / 'evaluation/recomputed_m6anet')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new benchmark output directory')

    # Reconstruct the original test membership from the fixed gene split.
    rows = build_manifest(args.manifest, args.labels0, args.labels1)
    rows = rows.loc[rows.split.eq('test'), ['sample_id'] + KEYS + ['gene_id', 'label']]
    pred = pd.read_csv(args.xgb_predictions, dtype={'sample_id': str, 'transcript_id': str, 'gene_id': str})

    # Require one XGBoost prediction for every test observation.
    if pred.sample_id.duplicated().any() or set(pred.sample_id) != set(rows.sample_id):
        raise ValueError('XGBoost predictions must cover exactly the original test')
    matched = rows.merge(pred, on='sample_id', validate='one_to_one', suffixes=('', '_pred'))

    # Check metadata as well as sample IDs to catch mismatched prediction files.
    for col in KEYS + ['gene_id', 'label']:
        if not matched[col].eq(matched[col + '_pred']).all():
            raise ValueError(f'XGBoost test metadata differs on {col}')

    parts = []
    for source, path in [('data0', args.m6anet_data0), ('data1', args.m6anet_data1)]:
        frame = pd.read_csv(path, dtype={'transcript_id': str})

        # Use site-level probabilities and retain the source dataset when joining.
        frame = frame.rename(columns={'probability_modified': 'm6anet_score'})
        frame['source_dataset'] = source
        parts.append(frame[KEYS + ['m6anet_score']])
    m6anet = pd.concat(parts, ignore_index=True)
    if m6anet.duplicated(KEYS).any():
        raise ValueError('Duplicate m6Anet source/site identifiers')

    # Record missing m6Anet predictions, then score both methods on shared sites.
    common = matched.merge(m6anet, on=KEYS, how='left', validate='one_to_one', indicator=True)
    missing = common.loc[common['_merge'].eq('left_only'), KEYS]
    common = common.loc[common['_merge'].eq('both')].copy()
    if common.empty:
        raise ValueError('No shared test sites')
    for col in ['xgb_score', 'm6anet_score']:
        common[col] = pd.to_numeric(common[col], errors='raise')
        if not np.isfinite(common[col]).all() or not common[col].between(0, 1).all():
            raise ValueError(f'Invalid scores: {col}')

    # Report combined metrics and separate results for each source dataset.
    results = []
    subsets = {'overall_common': common, **{s: f for s, f in common.groupby('source_dataset')}}
    for dataset, frame in subsets.items():
        for name, col in [('xgb_data0_data1', 'xgb_score'), ('m6anet_pretrained', 'm6anet_score')]:
            results.append({'model': name, 'dataset': dataset, **metrics(frame.label, frame[col])})

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / 'comparison_metrics.csv', index=False)
    missing.to_csv(out / 'missing_m6anet_sites.csv', index=False)

    # Save coverage counts alongside metrics so exclusions are visible.
    (out / 'coverage.json').write_text(json.dumps({'original_test': len(rows), 'common_sites': len(common),
                                                'missing_m6anet': len(missing),
                                                'm6anet_rows_outside_test': int((~m6anet.set_index(KEYS).index.isin(rows.set_index(KEYS).index)).sum())}, indent=2))
    print(pd.DataFrame(results).to_string(index=False))
    print(f'Shared test: {len(common):,}/{len(rows):,}. Pretrained m6Anet training overlap is unknown.')


if __name__ == '__main__':
    main()