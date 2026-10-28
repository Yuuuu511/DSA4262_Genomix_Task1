"""Fixed nine-signal-mean logistic baseline on the SAME joint gene partitions."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train import build_manifest, extract_selected, grouped_metrics, membership_hash, sha256
from xgb_features import FEATURE_NAMES, VERSION


def classifier():
    # Use only the first nine features, which are the signal means.
    # Keeping the scaler in the pipeline fits it only on the supplied training rows.
    return Pipeline([
        ('means', ColumnTransformer([('scale_nine_means', StandardScaler(), list(range(9)))], remainder='drop')),
        ('classifier', LogisticRegression(C=1, class_weight='balanced', max_iter=2000,
                                          solver='lbfgs', random_state=4262)),
    ])


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data0', type=Path, default=root.parent/'data0/dataset0.json.gz')
    parser.add_argument('--data1', type=Path, default=root.parent/'data1/dataset1.json.gz')
    parser.add_argument('--labels0', type=Path, default=root.parent/'data0/data.info.labelled')
    parser.add_argument('--labels1', type=Path, default=root.parent/'data1/data.info')
    parser.add_argument('--manifest', type=Path, default=root/'splits/gene_split.csv')
    parser.add_argument('--output-dir', type=Path, default=root/'artifacts/retrained_baseline')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()

    # Reuse the XGBoost gene partitions for a consistent comparison.
    manifest = build_manifest(args.manifest, args.labels0, args.labels1)
    paths = {'data0': args.data0, 'data1': args.data1}

    if args.evaluate_test:
        model_path = args.model or args.output_dir/'baseline.joblib'
        bundle = joblib.load(model_path)

        # Check that the saved model uses the same sites and feature code.
        if bundle.get('membership_sha256') != membership_hash(manifest):
            raise ValueError('Baseline membership differs from supplied split/labels')
        if bundle.get('feature_hash') != sha256(root/'xgb_features.py'):
            raise ValueError('Baseline feature implementation differs')

        out = args.output_dir/'evaluation'
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Baseline evaluation output already exists')

        # Evaluate the saved model on test sites without fitting it again.
        rows = manifest.loc[manifest.split.eq('test')].reset_index(drop=True)
        X = extract_selected(paths, rows)
        result = grouped_metrics(rows, bundle['model'].predict_proba(X)[:,1], {'model':'logistic_9means'})
        out.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(result).to_csv(out/'test_metrics.csv',index=False)
        print(pd.DataFrame(result).to_string(index=False))
        print('Fixed baseline reporting only; no tuning based on test.')
        return

    if args.model:
        parser.error('--model is for evaluation only')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Choose a new baseline output directory')

    # Test sites are excluded from both fits below.
    rows = manifest.loc[manifest.split.isin(['train','validation'])].reset_index(drop=True)
    X = extract_selected(paths, rows)
    train_mask, val_mask = rows.split.eq('train').to_numpy(), rows.split.eq('validation').to_numpy()
    y = rows.label.to_numpy()

    # Measure validation performance using a model fitted on training rows only.
    model = classifier().fit(X[train_mask], y[train_mask])
    result = grouped_metrics(rows.loc[val_mask].reset_index(drop=True),
                             model.predict_proba(X[val_mask])[:,1], {'model':'logistic_9means'})

    # Refit the fixed baseline and its scaler on train plus validation.
    final = classifier().fit(X,y)
    args.output_dir.mkdir(parents=True,exist_ok=True)

    # Save compatibility information alongside the fitted pipeline.
    bundle = {'model':final,'positive_weight':1,'feature_names':FEATURE_NAMES,'feature_version':VERSION,
              'feature_hash':sha256(root/'xgb_features.py'),'membership_sha256':membership_hash(manifest),
              'training_data':'data0+data1 train+validation; nine-mean logistic baseline'}
    joblib.dump(bundle,args.output_dir/'baseline.joblib',compress=3)
    pd.DataFrame(result).to_csv(args.output_dir/'validation_metrics.csv',index=False)
    (args.output_dir/'training_report.json').write_text(json.dumps({
        'method':'LogisticRegression C=1, class_weight=balanced, max_iter=2000, seed=4262',
        'inputs':'First nine signal-mean features only; scaling fitted inside training pipeline.',
        'selection':'Fixed simple baseline; no candidate tuning.',
        'training_scope':'data0+data1 train+validation; same gene split as XGBoost',
        'model_sha256':sha256(args.output_dir/'baseline.joblib'),
        'notes':['Validation metrics computed before train+validation refit.',
                 'Analytic XGBoost weight correction is not applied to this logistic baseline.',
                 'Neither baseline nor XGBoost probabilities are empirically calibrated.']},indent=2))
    print(pd.DataFrame(result).to_string(index=False))
    print('Baseline saved; test not evaluated.')


if __name__ == '__main__':
    main()