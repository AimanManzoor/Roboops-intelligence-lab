import json
import sys
import unittest
import tempfile
import subprocess

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class TestPipelineIntegration(unittest.TestCase):

    def test_end_to_end_pipeline(self):

        with tempfile.TemporaryDirectory() as directory:

            workspace = Path(directory)

            raw = workspace / "data/raw/robot.data/01-trainingdata"
            raw.mkdir(parents=True)

            for file_id in range(2):

                file = raw / f"sample_{file_id}.txt"

                with file.open("w") as f:

                    for i in range(40):

                        record = {
                            "timestamp": str(
                                datetime(2025, 1, 1)
                                + timedelta(seconds=i)
                            ),
                            "RobotJointData": {
                                "RightArm": [
                                    {
                                        "axis": axis,
                                        "speed": str(i + axis),
                                        "torq": str(i * 0.5),
                                        "angle": str(i + file_id)
                                    }
                                    for axis in range(1, 8)
                                ]
                            },
                            "Energy": None
                        }

                        f.write(json.dumps(record) + "\n")

            for script in [
                "data_ingestion.py",
                "feature_engineering.py"
            ]:

                subprocess.run(
                    [sys.executable, str(ROOT / "src" / script)],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True
                )

            raw_df = pd.read_parquet(
                workspace / "data/processed/training_sample.parquet"
            )

            features = pd.read_parquet(
                workspace / "data/processed/training_features.parquet"
            )

            self.assertEqual(raw_df.shape, (80, 23))
            self.assertEqual(features.shape, (22, 128))
            self.assertEqual(features.isna().sum().sum(), 0)
            self.assertEqual(features["source_file"].nunique(), 2)

            numeric = features.select_dtypes(include="number")

            self.assertTrue(
                np.isfinite(numeric.to_numpy()).all()
            )

            for _, group in features.groupby("source_file"):

                self.assertEqual(len(group), 11)

                self.assertAlmostEqual(
                    group.iloc[0]["axis_1_speed_mean"],
                    15.5
                )


if __name__ == "__main__":
    unittest.main()
