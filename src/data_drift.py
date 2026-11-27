from pathlib import Path

import json
import numpy as np
import pandas as pd

from itertools import islice

TRAIN = Path("data/processed/training_sample.parquet")
TEST = Path("data/raw/robot.data/02-testdata")

SAMPLE_LIMIT = 10000

train = pd.read_parquet(TRAIN)

results = []

for file in sorted(TEST.glob("*.txt")):

    rows = []

    with file.open("r", encoding="utf-8") as f:

        for line in islice(f, SAMPLE_LIMIT):

            try:
                record = json.loads(line)

                joints = record["RobotJointData"]["RightArm"]

                if sorted(j["axis"] for j in joints) != list(range(1, 8)):
                    continue

                row = {}

                for joint in joints:

                    axis = joint["axis"]

                    row[f"axis_{axis}_speed"] = float(joint["speed"])
                    row[f"axis_{axis}_torque"] = float(joint["torq"])
                    row[f"axis_{axis}_angle"] = float(joint["angle"])

                rows.append(row)

            except (ValueError, KeyError, TypeError):
                continue

    test = pd.DataFrame(rows)

    if test.empty:
        continue

    test = test.replace([np.inf, -np.inf], np.nan)

    for column in test.columns:

        train_mean = train[column].mean()
        train_std = train[column].std()

        test_mean = test[column].mean()

        standardized_shift = (
            abs(test_mean - train_mean) / train_std
            if train_std > 0 else np.nan
        )

        results.append({
            "file": file.name,
            "feature": column,
            "sample_size": test[column].count(),
            "train_mean": train_mean,
            "test_mean": test_mean,
            "standardized_shift": standardized_shift
        })

report = pd.DataFrame(results)

if report.empty:
    raise RuntimeError("No valid test observations found.")

report.to_csv(
    "data/processed/data_drift_report.csv",
    index=False
)

print("\n--- DATA DRIFT REPORT ---")

print(
    report.sort_values(
        "standardized_shift",
        ascending=False
    ).head(15).round(4).to_string(index=False)
)

print("\nComparisons:", len(report))
