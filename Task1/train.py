"""Train data0+data1 XGBoost on the existing fixed gene manifest.

Place beside the unchanged xgb_features.py and predict.py in task1/.
Default action trains; --check-only validates metadata; --evaluate-test performs
separate reporting on the frozen saved model without any fitting or selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier
from xgb_features import FEATURE_NAMES, VERSION, iter_sites, site_features
from prepare_split import eligible_metadata, COLUMNS

# Include the dataset name because the same site can appear in both sources.
KEYS = ['source_dataset', 'transcript_id', 'transcript_position']
# Try four combinations of tree depth and positive-class weight.
CANDIDATES = [(3, 1), (5, 1), (3, 4), (5, 4)]
TRAINING_SCOPE = 'data0+data1 train+validation; fixed gene manifest'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def feature_hash():
    return hashlib.sha256(Path(__file__).with_name('xgb_features.py').read_bytes()).hexdigest()


def read_manifest(path):
    frame = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(
        path, dtype={'sample_id': str, 'source_dataset': str, 'transcript_id': str, 'gene_id': str})
    required = ['sample_id'] + KEYS + ['gene_id', 'label', 'split']
    if not set(required).issubset(frame.columns):
        raise ValueError(f'Manifest requires {required}')
    if frame[required].isna().any().any():
        raise ValueError('Missing manifest values')
    if frame.sample_id.duplicated().any() or frame.duplicated(KEYS).any():
        raise ValueError('Duplicate sample or source/site identifiers')
    if set(frame.split) != {'train', 'validation', 'test'}:
        raise ValueError('Expected unchanged train/validation/test split')
    if not frame.source_dataset.isin(['data0', 'data1']).all():
        raise ValueError('Expected data0/data1 only')
    if not frame.label.isin([0, 1]).all():
        raise ValueError('Binary labels required')
    # Keep related sites in one split to avoid leakage between partitions.
    if frame.groupby('gene_id').split.nunique().max() > 1:
        raise ValueError('A gene occurs in multiple splits')
    if frame.groupby('transcript_id').gene_id.nunique().max() > 1:
        raise ValueError('A transcript maps to different genes')
    if frame.groupby('transcript_id').split.nunique().max() > 1:
        raise ValueError('A transcript occurs in multiple splits')
    if frame.groupby(['transcript_id', 'transcript_position']).split.nunique().max() > 1:
        raise ValueError('A site occurs in multiple splits')
    positions = pd.to_numeric(frame.transcript_position, errors='raise')
    if not np.isfinite(positions).all() or (positions % 1 != 0).any():
        raise ValueError('Positions must be finite integers')
    frame['transcript_position'] = positions.astype(np.int64)
    return frame


def build_manifest(split_file, labels0, labels1):
    """Reconstruct site membership from local labels and public gene assignments."""
    genes = pd.read_csv(split_file, dtype={'gene_id': str, 'split': str})
    if genes.columns.tolist() != ['gene_id', 'split'] or genes.isna().any().any():
        raise ValueError('Split file requires gene_id,split; no site labels are bundled')
    if genes.gene_id.duplicated().any() or set(genes.split) != {'train', 'validation', 'test'}:
        raise ValueError('Gene assignments must be unique and include train/validation/test')
    # Rebuild site membership locally; the public split file contains no labels.
    rows, excluded = eligible_metadata(labels0, labels1)
    rows = rows.merge(genes, on='gene_id', how='left', validate='many_to_one')
    if rows.split.isna().any():
        raise ValueError('Input genes are absent from the fixed split; generate a new split explicitly')
    manifest = pd.concat([rows.loc[rows.split.eq(s)] for s in ['train', 'validation', 'test']],
                         ignore_index=True)[COLUMNS]
    return read_manifest(manifest)


def membership_hash(manifest):
    columns = ['sample_id', 'source_dataset', 'gene_id', 'transcript_id', 'transcript_position', 'split']
    stable = manifest[columns].sort_values('sample_id').to_csv(index=False)
    return hashlib.sha256(stable.encode()).hexdigest()


def extract_selected(paths, rows):
    """Compute features only for requested rows, keeping manifest order."""
    # Map each requested site to its row so features stay aligned with labels.
    wanted = {key: i for i, key in enumerate(rows[KEYS].itertuples(index=False, name=None))}
    if len(wanted) != len(rows):
        raise ValueError('Duplicate requested sites')
    X = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    seen = set()
    for source in ['data0', 'data1']:
        if not rows.source_dataset.eq(source).any():
            continue
        path = Path(paths[source])
        print(f'Extracting selected XGBoost features from {source}...', flush=True)
        for count, (transcript, position, sequence, reads) in enumerate(iter_sites(path), 1):
            key = (source, transcript, position)
            if key in wanted:
                if key in seen:
                    raise ValueError(f'Duplicate raw site: {key}')
                X[wanted[key]] = site_features(sequence, reads)
                seen.add(key)
            if count % 20000 == 0:
                print(f'  Scanned {count:,} sites', flush=True)
    if seen != set(wanted):
        raise ValueError(f'Missing raw features for {len(set(wanted) - seen)} sites')
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite feature values')
    return X


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both classes required for evaluation')
    if len(p) != len(y) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probability scores')
    # Report trapezoidal PR area and average precision as separate metrics.
    precision, recall, _ = precision_recall_curve(y, p)
    return {'n_sites': len(y), 'positive_rate': float(y.mean()),
            'roc_auc': float(roc_auc_score(y, p)), 'pr_auc': float(auc(recall, precision)),
            'average_precision': float(average_precision_score(y, p))}


def probabilities(model, X, weight):
    p = model.predict_proba(X)[:, 1].astype(float)
    # Same analytic class-weight correction as the original XGBoost method.
    return p / (weight - (weight - 1) * p)


def model_parameters(depth, weight, args):
    return dict(max_depth=depth, scale_pos_weight=weight, learning_rate=.05,
                n_estimators=args.max_rounds, min_child_weight=15, subsample=.85,
                colsample_bytree=.85, reg_lambda=10, reg_alpha=.1, tree_method='hist',
                n_jobs=args.n_jobs, random_state=4262, eval_metric='aucpr',
                early_stopping_rounds=args.early_stopping_rounds)


def read_labels(path):
    labels = pd.read_csv(path, dtype={'transcript_id': str, 'gene_id': str})
    keys = ['transcript_id', 'transcript_position']
    required = keys + ['label']
    if not set(required).issubset(labels.columns) or labels[required].isna().any().any():
        raise ValueError('Labels require transcript_id, transcript_position, label')
    positions = pd.to_numeric(labels.transcript_position, errors='raise')
    if not np.isfinite(positions).all() or (positions % 1 != 0).any():
        raise ValueError('Label positions must be finite integers')
    labels['transcript_position'] = positions.astype(np.int64)
    if labels.duplicated(keys).any() or not labels.label.isin([0, 1]).all():
        raise ValueError('Labels must be unique and binary')
    return labels


def validate_labels(paths, manifest):
    """Check label identity and gene mapping; never add sites outside the manifest."""
    tables = {source: read_labels(path) for source, path in paths.items()}
    pairs = []
    for table in tables.values():
        if 'gene_id' in table:
            pairs.append(table[['transcript_id', 'gene_id']].dropna().drop_duplicates())
    if not pairs:
        raise ValueError('A supplied label table must contain gene_id (normally data0).')
    mapping = pd.concat(pairs, ignore_index=True).drop_duplicates()
    if mapping.groupby('transcript_id').gene_id.nunique().max() > 1:
        raise ValueError('Conflicting gene mappings in input labels')
    # Use the available transcript-to-gene mappings to check both datasets.
    gene_map = mapping.drop_duplicates('transcript_id').set_index('transcript_id').gene_id
    keys = ['transcript_id', 'transcript_position']
    audit = []
    for source, labels in tables.items():
        expected = manifest.loc[manifest.source_dataset.eq(source)]
        matched = expected.merge(labels[keys + ['label']], on=keys, how='left',
                                 validate='one_to_one', suffixes=('', '_input'), indicator=True)
        if not matched['_merge'].eq('both').all() or not matched.label.eq(matched.label_input).all():
            raise ValueError(f'{source}: input labels missing or disagree with manifest')
        mapped = expected.transcript_id.map(gene_map)
        if mapped.isna().any() or not mapped.eq(expected.gene_id).all():
            raise ValueError(f'{source}: input gene mapping disagrees with manifest')
        audit.append({'source_dataset': source, 'label_sites': len(labels),
                      'retained_manifest_sites': len(expected),
                      'label_sites_outside_manifest': len(labels) - len(expected)})
    return audit


def grouped_metrics(rows, predictions, extra=None):
    result = []
    for dataset in ['overall', 'data0', 'data1']:
        mask = np.ones(len(rows), dtype=bool) if dataset == 'overall' else rows.source_dataset.eq(dataset).to_numpy()
        result.append({**(extra or {}), 'dataset': dataset,
                       **metrics(rows.loc[mask, 'label'], predictions[mask])})
    return result


def evaluate(args, manifest, signature):
    # Compatibility checks are deliberately local: old evaluate_test.py only accepts data0-only.
    from predict import load_model
    model_path = args.model or args.output_dir / 'selected_xgb.joblib'
    bundle = load_model(model_path)
    if bundle.get('training_data') != TRAINING_SCOPE:
        raise ValueError('This evaluator expects the joint data0+data1 training model')
    if bundle.get('membership_sha256') != membership_hash(manifest) or bundle.get('manifest_sha256') != signature:
        raise ValueError('Model and evaluation manifests differ')
    out = args.evaluation_dir or args.output_dir / 'evaluation'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Evaluation output directory is nonempty; choose a new --evaluation-dir')
    # This mode only scores the saved model on test sites; it does not fit it.
    rows = manifest.loc[manifest.split.eq('test')].reset_index(drop=True)
    X = extract_selected({'data0': args.data0, 'data1': args.data1}, rows)
    p = probabilities(bundle['model'], X, bundle['positive_weight'])
    results = grouped_metrics(rows, p, {'model': 'xgb_data0_data1'})
    if sha256(args.manifest) != signature:
        raise ValueError('Manifest changed during evaluation')
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / 'test_metrics.csv', index=False)
    rows.assign(xgb_score=p).to_csv(out / 'heldout_test_predictions.csv', index=False)
    (out / 'evaluation_report.json').write_text(json.dumps({
        'model_path': str(model_path.resolve()), 'model_sha256': sha256(model_path),
        'manifest_sha256': signature, 'no_fitting_or_selection': True,
        'notes': ['Original test membership retained. Reporting only; do not retune using these results.',
                  'These test partitions have been inspected in earlier experiments.']}, indent=2))
    print(pd.DataFrame(results).to_string(index=False))
    print('Test reporting only. Do not use these results for tuning or choosing a split.')


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data0', type=Path, default=root.parent / 'data0/dataset0.json.gz')
    parser.add_argument('--data1', type=Path, default=root.parent / 'data1/dataset1.json.gz')
    parser.add_argument('--labels0', type=Path, default=root.parent / 'data0/data.info.labelled')
    parser.add_argument('--labels1', type=Path, default=root.parent / 'data1/data.info')
    parser.add_argument('--manifest', type=Path, default=root / 'splits/gene_split.csv',
                        help='Label-free gene_id,split assignments')
    parser.add_argument('--output-dir', type=Path, default=root / 'artifacts/retrained')
    parser.add_argument('--max-rounds', type=int, default=1200)
    parser.add_argument('--early-stopping-rounds', type=int, default=60)
    parser.add_argument('--n-jobs', type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Validate metadata and splits without training')
    mode.add_argument('--evaluate-test', action='store_true', help='Separate frozen-model reporting, no training')
    parser.add_argument('--model', type=Path, help='Saved joint model, for --evaluate-test only')
    parser.add_argument('--evaluation-dir', type=Path)
    args = parser.parse_args()
    if min(args.max_rounds, args.early_stopping_rounds, args.n_jobs) < 1:
        parser.error('Round and worker counts must be positive')
    if (args.model or args.evaluation_dir) and not args.evaluate_test:
        parser.error('--model and --evaluation-dir are for --evaluate-test only')
    signature = sha256(args.manifest)
    manifest = build_manifest(args.manifest, args.labels0, args.labels1)
    for source in ['data0', 'data1']:
        for split in ['train', 'validation', 'test']:
            labels = manifest.loc[manifest.source_dataset.eq(source) & manifest.split.eq(split), 'label']
            if labels.nunique() != 2:
                raise ValueError(f'{source} {split} requires both classes')
    if args.evaluate_test:
        evaluate(args, manifest, signature)
        return
    audit = validate_labels({'data0': args.labels0, 'data1': args.labels1}, manifest)
    print('Label/manifest coverage (outside-manifest sites remain excluded):', flush=True)
    print(pd.DataFrame(audit).to_string(index=False), flush=True)
    print('Manifest source/split counts:', flush=True)
    print(manifest.groupby(['source_dataset', 'split']).size().to_string(), flush=True)
    if args.check_only:
        print('Checks passed: shared gene partitions, label identity and source coverage. No training performed.')
        return
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output folder is nonempty; use a new folder')
    # Leave test sites out of feature extraction for training and model selection.
    rows = manifest.loc[manifest.split.isin(['train', 'validation'])].reset_index(drop=True)
    print(rows.groupby(['source_dataset', 'split']).size().to_string(), flush=True)
    features_signature = feature_hash()
    X = extract_selected({'data0': args.data0, 'data1': args.data1}, rows)
    y = rows.label.to_numpy().astype(int)
    train = rows.split.eq('train').to_numpy()
    val = rows.split.eq('validation').to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    results, detailed = [], []
    best = None
    best_validation_predictions = None
    for depth, weight in CANDIDATES:
        model = XGBClassifier(**model_parameters(depth, weight, args))
        print(f'Training depth={depth}, positive_weight={weight}...', flush=True)
        # Use validation PR area for early stopping, not the test set.
        model.fit(X[train], y[train], eval_set=[(X[val], y[val])], verbose=False)
        p = probabilities(model, X[val], weight)
        m = metrics(y[val], p)
        result = dict(max_depth=depth, positive_weight=weight,
                      boosting_rounds=int(model.best_iteration + 1),
                      selection_score=(m['roc_auc'] + m['pr_auc']) / 2, **m)
        results.append(result)
        detailed.extend(grouped_metrics(rows.loc[val].reset_index(drop=True), p,
                        {'max_depth': depth, 'positive_weight': weight,
                         'boosting_rounds': result['boosting_rounds']}))
        print(json.dumps(result), flush=True)
        pd.DataFrame(results).to_csv(out / 'validation_metrics.csv', index=False)
        pd.DataFrame(detailed).to_csv(out / 'validation_metrics_by_dataset.csv', index=False)
        # Choose the candidate using the average validation ROC-AUC and PR-AUC.
        if best is None or result['selection_score'] > best['selection_score']:
            best = result.copy()
            best_validation_predictions = p.copy()
    params = model_parameters(best['max_depth'], best['positive_weight'], args)
    # Reuse the selected round count when fitting on train and validation together.
    params.pop('early_stopping_rounds')
    params['n_estimators'] = best['boosting_rounds']
    if not rows.split.isin(['train', 'validation']).all() or set(rows.source_dataset) != {'data0', 'data1'}:
        raise ValueError('Unexpected training membership')
    print('Selected configuration:', json.dumps(best), flush=True)
    print('Refitting on data0+data1 train+validation only...', flush=True)
    final = XGBClassifier(**params).fit(X, y)
    if sha256(args.manifest) != signature or feature_hash() != features_signature:
        raise ValueError('Manifest or feature implementation changed during training')
    # Save the feature and split hashes so later runs can check compatibility.
    bundle = {'model': final, 'positive_weight': best['positive_weight'],
              'feature_names': FEATURE_NAMES, 'feature_version': VERSION,
              'feature_hash': features_signature, 'membership_sha256': membership_hash(manifest),
              'manifest_sha256': signature, 'training_data': TRAINING_SCOPE, 'seed': 4262}
    joblib.dump(bundle, out / 'selected_xgb.joblib', compress=3)
    report = {'selected': best, 'candidates': results, 'parameters': params,
              'training_scope': 'data0+data1 train+validation only',
              'selection_rule': '(combined validation ROC-AUC + combined validation trapezoidal PR-AUC)/2',
              'manifest': str(args.manifest.resolve()),
              'manifest_sha256': signature, 'feature_hash': features_signature,
              'model_sha256': sha256(out / 'selected_xgb.joblib'),
              'label_coverage': audit,
              'split_counts': [{'source_dataset': source, 'split': split, 'n_sites': int(n)}
                               for (source, split), n in manifest.groupby(['source_dataset', 'split']).size().items()],
              'input_paths': {'data0': str(args.data0.resolve()), 'data1': str(args.data1.resolve()),
                              'labels0': str(args.labels0.resolve()), 'labels1': str(args.labels1.resolve())},
              'script_sha256': sha256(__file__),
              'versions': {'xgboost': xgboost.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__},
              'notes': ['Test excluded from all feature extraction, fitting and selection.',
                        'Input labels, including test identities, checked against manifest only; test not used for fitting/selection.',
                        'Validation used for early stopping (aucpr) and combined configuration selection.',
                        'PR-AUC is trapezoidal area; AP is reported separately.',
                        'Combined metrics weight each source/site observation once; inspect per-source validation too.',
                        'Unmapped/excluded data1 sites are not added to the existing manifest.',
                        'Neither data2 nor the four new unlabelled datasets are used.',
                        'Model is NOT refitted on test genes.',
                        'Existing data0-only model was tuned using data0 validation only: not a controlled training-source ablation.',
                        'Original test has been inspected previously; final reporting is exploratory.']}
    (out / 'training_report.json').write_text(json.dumps(report, indent=2))
    # Locally generated label-bearing tables are not part of the public package.
    manifest.to_csv(out / 'private_split_manifest.csv', index=False)
    rows.loc[val].assign(xgb_score=best_validation_predictions).to_csv(
        out / 'selected_validation_predictions.csv', index=False)
    print('Selected validation metrics (before final refit):', flush=True)
    selected_details = [row for row in detailed if row['max_depth'] == best['max_depth'] and row['positive_weight'] == best['positive_weight']]
    print(pd.DataFrame(selected_details).to_string(index=False), flush=True)
    print(f'Saved {out / "selected_xgb.joblib"}. Test was not evaluated.')


if __name__ == '__main__':
    main()
