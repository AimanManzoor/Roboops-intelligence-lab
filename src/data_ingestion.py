import json
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/robot.data/01-trainingdata")
OUTPUT_PATH = Path("data/processed/training_sample.parquet")
SAMPLE_LIMIT = 10000

records = []
rejected = 0

for file in sorted(DATA_PATH.glob("*.txt")):
    with file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)

                row = {
                    "timestamp": pd.to_datetime(
                        data["timestamp"], errors="raise"
                    ),
                    "source_file": file.name
                }

                joints = data["RobotJointData"]["RightArm"]

                if sorted(j["axis"] for j in joints) != list(range(1, 8)):
                    rejected += 1
                    continue

                for joint in joints:
                    axis = joint["axis"]

                    row[f"axis_{axis}_speed"] = float(joint["speed"])
                    row[f"axis_{axis}_torque"] = float(joint["torq"])
                    row[f"axis_{axis}_angle"] = float(joint["angle"])

                records.append(row)

                if len(records) >= SAMPLE_LIMIT:
                    break

            except (ValueError, TypeError, KeyError):
                rejected += 1

    if len(records) >= SAMPLE_LIMIT:
        break

if not records:
    raise RuntimeError("No valid records found.")

df = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_parquet(
    OUTPUT_PATH,
    index=False,
    compression="snappy"
)

print("\n--- INGESTION REPORT ---")
print("Dataset shape:", df.shape)
print("Rejected records:", rejected)
print("Missing values:", df.isna().sum().sum())
print("Memory usage (MB):", round(
    df.memory_usage(deep=True).sum() / 1024**2, 2
))
print("Output:", OUTPUT_PATH)
print(df.head())
