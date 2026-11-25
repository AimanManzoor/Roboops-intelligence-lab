import json
from pathlib import Path
from collections import Counter

DATA_PATH = Path("data/raw/robot.data/01-trainingdata")
SAMPLE_LIMIT = 10000

fields = Counter()
axes = Counter()
invalid_records = 0
missing_energy = 0
total_records = 0

for file in DATA_PATH.glob("*.txt"):
    print(f"\nInspecting: {file.name}")

    with file.open("r", encoding="utf-8") as f:
        for line in f:
            if total_records >= SAMPLE_LIMIT:
                break

            try:
                record = json.loads(line)

                if not isinstance(record, dict):
                    invalid_records += 1
                    continue

                total_records += 1
                fields.update(record.keys())

                if record.get("Energy") is None:
                    missing_energy += 1

                joint_data = record.get("RobotJointData", {})

                for joint in joint_data.get("RightArm", []):
                    axes[joint.get("axis")] += 1

            except (json.JSONDecodeError, AttributeError, TypeError):
                invalid_records += 1

print("\n--- DATA QUALITY REPORT ---")
print("Valid records:", total_records)
print("Invalid records:", invalid_records)
print("Missing energy:", missing_energy)
print("Available fields:", dict(fields))
print("Axis observations:", dict(axes))
