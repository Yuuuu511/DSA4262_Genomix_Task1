"""Small executable prediction-contract checks; no real retraining."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from predict import predict_file, load_model
from validate_submission import validate
from xgb_features import site_features

ROOT = Path(__file__).resolve().parent
MODEL = ROOT/'artifacts/selected_xgb.joblib'


class PipelineTests(unittest.TestCase):
    def test_default_joint_model_and_label_free_metadata(self):
        # Check the model's training scope and matching gene split.
        bundle = load_model(MODEL)
        self.assertEqual(bundle['training_data'], 'data0+data1 train+validation; fixed gene manifest')

        # The saved bundle should not contain the private site-level manifest.
        self.assertNotIn('manifest',bundle)
        expected = hashlib.sha256((ROOT/'splits/gene_split.csv').read_bytes()).hexdigest()
        self.assertEqual(bundle['manifest_sha256'], expected)

    def test_real_example_and_batch_invariance(self):
        # Temporary outputs are removed automatically after the test.
        with tempfile.TemporaryDirectory() as directory:
            # Predictions should agree across different batch sizes.
            for size in [1,7,2048]:
                output = Path(directory)/f'p{size}.csv'
                self.assertEqual(predict_file(ROOT/'example/test.json.gz',MODEL,output,size),20)
                self.assertEqual(validate(ROOT/'example/test.json.gz',output)['status'],'PASS')
                actual=pd.read_csv(output)
                expected=pd.read_csv(ROOT/'example/expected_predictions.csv')

                # Match site identifiers exactly and allow small rounding differences in scores.
                pd.testing.assert_frame_equal(actual.iloc[:,:2],expected.iloc[:,:2])
                np.testing.assert_allclose(actual.score,expected.score,atol=1e-6,rtol=0)

    def test_plain_json_position_zero_duplicates_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            raw=directory/'sample.json'; output=directory/'output.csv'

            # Use a small plain JSON input to check that position zero is accepted.
            record={'tx_id_0':{'0':{'AAGACCA':[[.01,2,100,.02,3,110,.03,4,90]]*3}}}
            raw.write_text(json.dumps(record)+'\n')
            predict_file(raw,MODEL,output)
            self.assertEqual(validate(raw,output)['rows'],1)
            self.assertEqual(pd.read_csv(output).transcript_position.iloc[0],0)

            # Existing output files should not be overwritten by default.
            with self.assertRaises(FileExistsError):predict_file(raw,MODEL,output)

            # Duplicate input sites should fail without leaving a final output file.
            raw.write_text(json.dumps(record)+'\n'+json.dumps(record)+'\n')
            with self.assertRaises(ValueError):predict_file(raw,MODEL,directory/'duplicate.csv')
            self.assertFalse((directory/'duplicate.csv').exists())

    def test_invalid_reads(self):
        # Reject empty reads, the wrong feature count and nonfinite values.
        for reads in [[],[[1]*8],[[float('nan')]*9]]:
            with self.assertRaises(ValueError):site_features('AAGACCA',reads)


if __name__=='__main__':
    unittest.main()