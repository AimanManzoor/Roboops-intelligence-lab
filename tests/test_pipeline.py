import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestRoboOpsPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_parquet(
            "data/processed/training_sample.parquet"
        )

        cls.features = pd.read_parquet(
            "data/processed/training_features.parquet"
        )

    def test_raw_schema(self):
        self.assertEqual(self.raw.shape[1], 23)

    def test_feature_schema(self):
        self.assertEqual(self.features.shape[1], 128)

    def test_missing_values(self):
        self.assertEqual(
            self.features.isna().sum().sum(), 0
        )

    def test_finite_values(self):
        numeric = self.features.select_dtypes(
            include="number"
        )

        self.assertTrue(
            np.isfinite(numeric.to_numpy()).all()
        )

    def test_temporal_order(self):
        self.assertTrue(
            self.features["timestamp"].is_monotonic_increasing
        )

    def test_duplicate_timestamps(self):
        self.assertFalse(
            self.features.duplicated(
                subset=["source_file", "timestamp"]
            ).any()
        )

    def test_output_exists(self):
        self.assertTrue(
            Path("data/processed/training_features.parquet").exists()
        )


if __name__ == "__main__":
    unittest.main()

