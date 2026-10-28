"""Create a model-independent gene-level split directly from label metadata.

Supply data0+data1 labels to reproduce the current joint-source manifest; data1
uses data0's transcript-to-gene map where its gene mapping is missing. Both
sources' train/validation partitions are used by train.py. Public output contains
gene_id,split assignments only; training labels stay in the local input files.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SITE_KEYS = ['transcript_id', 'transcript_position']
COLUMNS = ['sample_id', 'source_dataset', 'gene_id'] + SITE_KEYS + ['label', 'split']


def load_labels(path, source):
    frame = pd.read_csv(path, dtype={'transcript_id': str, 'gene_id': 'string'})
    frame.columns = frame.columns.str.strip()
    if not set(SITE_KEYS + ['label']).issubset(frame.columns):
        raise ValueError(f'{source}: labels require transcript_id, transcript_position, label')
    if frame[SITE_KEYS].isna().any().any():
        raise ValueError(f'{source}: missing site identifiers')
    positions = pd.to_numeric(frame.transcript_position, errors='raise')
    if not np.isfinite(positions).all() or (positions % 1 != 0).any():
        raise ValueError(f'{source}: invalid integer positions')
    frame['transcript_position'] = positions.astype(np.int64)
    frame['label'] = pd.to_numeric(frame.label, errors='raise')
    if not frame.label.isin([0, 1]).all():
        raise ValueError(f'{source}: binary labels required')
    frame['label'] = frame.label.astype(int)
    if frame.duplicated(SITE_KEYS).any():
        raise ValueError(f'{source}: duplicate sites')
    if source == 'data0' and 'gene_id' not in frame:
        raise ValueError('data0 labels must contain gene_id')
    keep = SITE_KEYS + ['label'] + (['gene_id'] if 'gene_id' in frame else [])
    return frame[keep].copy()


def eligible_metadata(data0_labels, data1_labels=None):
    data0 = load_labels(data0_labels, 'data0')
    pairs = data0[['transcript_id', 'gene_id']].dropna().drop_duplicates()
    if pairs.transcript_id.duplicated().any():
        raise ValueError('A data0 transcript maps to multiple genes')
    # Use data0 as the reference for missing transcript-to-gene mappings.
    gene_map = pairs.set_index('transcript_id').gene_id
    parts = []
    excluded = {}
    for source, frame in [('data0', data0)] + (
            [('data1', load_labels(data1_labels, 'data1'))] if data1_labels else []):
        frame = frame.copy()
        # Keep existing gene IDs and fill missing ones from the reference map.
        if 'gene_id' not in frame:
            frame['gene_id'] = frame.transcript_id.map(gene_map)
        else:
            frame['gene_id'] = frame.gene_id.fillna(frame.transcript_id.map(gene_map))
        frame['gene_id'] = frame.gene_id.astype('string').str.strip().replace('', pd.NA)
        # Count and exclude sites without a gene ID because they cannot be grouped reliably.
        excluded[source] = int(frame.gene_id.isna().sum())
        frame = frame.loc[frame.gene_id.notna()].copy()
        # Reproduce the ordering of the original pandas outer JSON/label merge.
        frame = frame.sort_values(SITE_KEYS, kind='stable').reset_index(drop=True)
        frame['source_dataset'] = source
        # Include the source so shared sites in data0 and data1 have distinct sample IDs.
        frame['sample_id'] = source + ':' + frame.transcript_id + ':' + frame.transcript_position.astype(str)
        parts.append(frame)
    rows = pd.concat(parts, ignore_index=True)
    if rows.empty or rows.sample_id.duplicated().any():
        raise ValueError('No eligible sites or duplicate identifiers')
    if rows.groupby('transcript_id').gene_id.nunique().max() > 1:
        raise ValueError('Inconsistent transcript-to-gene mappings across sources')
    return rows, excluded


def make_manifest(rows, seed=42, test_fraction=.20, validation_fraction=.25):
    # Hold out test genes first; all sites from a gene stay together.
    outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    dev_idx, test_idx = next(outer.split(rows, groups=rows.gene_id))
    dev = rows.iloc[dev_idx]
    # Select validation genes from the remaining genes, using a second fixed seed.
    inner = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed + 1)
    train_idx, val_idx = next(inner.split(dev, groups=dev.gene_id))
    partitions = [('train', dev.iloc[train_idx]), ('validation', dev.iloc[val_idx]),
                  ('test', rows.iloc[test_idx])]
    manifest = pd.concat([frame.assign(split=split) for split, frame in partitions], ignore_index=True)[COLUMNS]
    # Check that neither a gene nor a shared site appears in multiple partitions.
    if manifest.groupby('gene_id').split.nunique().max() > 1:
        raise ValueError('A gene crosses splits')
    if manifest.groupby(SITE_KEYS).split.nunique().max() > 1:
        raise ValueError('A site crosses splits')
    for split, frame in partitions:
        if frame.label.nunique() != 2:
            raise ValueError(f'{split} needs both classes; do not choose splits by test performance')
        subset = frame.loc[frame.source_dataset.eq('data0')]
        if subset.label.nunique() != 2:
            raise ValueError(f'data0 {split} needs both classes')
    return manifest


def check_raw_coverage(path, labels, source):
    # Optional input check; no feature aggregation and no model training.
    from xgb_features import iter_sites
    expected = set(labels[SITE_KEYS].itertuples(index=False, name=None))
    seen = set()
    for tx, pos, _, _ in iter_sites(path):
        key = (tx, pos)
        if key in seen:
            raise ValueError(f'{source}: duplicate raw site {key}')
        seen.add(key)
    if seen != expected:
        raise ValueError(f'{source}: raw/label coverage differs: missing={len(expected-seen)}, extra={len(seen-expected)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data0-labels', type=Path, required=True)
    parser.add_argument('--data1-labels', type=Path)
    parser.add_argument('--data0-json', type=Path)
    parser.add_argument('--data1-json', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-against', type=Path, help='Require exact gene assignment reproduction')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-fraction', type=float, default=.20)
    parser.add_argument('--validation-fraction-of-remaining', type=float, default=.25)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1 or not 0 < args.validation_fraction_of_remaining < 1:
        parser.error('Split fractions must be between zero and one')
    if args.data1_json and not args.data1_labels:
        parser.error('--data1-json requires --data1-labels')
    if args.output.exists():
        raise FileExistsError('Output manifest exists; choose a new path')
    for source, raw, labels in [('data0', args.data0_json, args.data0_labels),
                               ('data1', args.data1_json, args.data1_labels)]:
        if raw:
            check_raw_coverage(raw, load_labels(labels, source), source)
    rows, excluded = eligible_metadata(args.data0_labels, args.data1_labels)
    manifest = make_manifest(rows, args.seed, args.test_fraction, args.validation_fraction_of_remaining)
    # Compare gene assignments rather than CSV row order when reproducing a split.
    if args.check_against:
        reference = pd.read_csv(args.check_against, dtype={'gene_id': 'string'})
        actual = manifest[['gene_id','split']].drop_duplicates().sort_values('gene_id').reset_index(drop=True)
        reference = reference[['gene_id','split']].drop_duplicates().sort_values('gene_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, reference, check_dtype=False)
        print('PASS: exactly reproduces fixed gene assignments.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Export only gene assignments; keep site labels in the local input files.
    manifest[['gene_id','split']].drop_duplicates().sort_values('gene_id').to_csv(args.output, index=False)
    print(manifest.groupby(['source_dataset', 'split']).size().to_string())
    print('Excluded unmapped genes:', json.dumps(excluded))
    print(f'Saved {args.output.resolve()}')


if __name__ == '__main__':
    main()
