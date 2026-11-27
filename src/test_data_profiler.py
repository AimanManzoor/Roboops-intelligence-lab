import json
from pathlib import Path
from collections import Counter
from itertools import islice

import pandas as pd

DATA_PATH = Path("data/raw/robot.data/02-testdata")
SAMPLE_LIMIT = 10000

results = []

for file in sorted(DATA_PATH.glob("*.txt")):

    counts = Counter()

    with file.open("r", encoding="utf-8") as f:

        for line in islice(f, SAMPLE_LIMIT):

            counts["sampled"] += 1

            try:
                record = json.loads(line)

                if not isinstance(record, dict):
                    counts["invalid"] += 1
                    continue

                counts["valid"] += 1

                energy = record.get("Energy")

                if isinstance(energy, dict):

                    for field in ["power", "voltage", "current"]:

                        if energy.get(field) is not None:
                            counts[field] += 1

                else:
                    counts["missing_energy"] += 1

            except json.JSONDecodeError:
                counts["invalid"] += 1

    results.append({
        "file": file.name,
        "sampled": counts["sampled"],
        "valid": counts["valid"],
        "invalid": counts["invalid"],
        "power_available": counts["power"],
        "voltage_available": counts["voltage"],
        "current_available": counts["current"],
        "missing_energy": counts["missing_energy"]
    })

report = pd.DataFrame(results)

print("\n--- TEST DATA PROFILE ---")
print(report.to_string(index=False))

report.to_csv(
    "data/processed/test_data_profile.csv",
    index=False
)
