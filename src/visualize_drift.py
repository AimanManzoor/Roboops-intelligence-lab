from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/data_drift_report.csv")

matrix = df.pivot(
    index="file",
    columns="feature",
    values="standardized_shift"
)

output = Path("docs/drift")
output.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(20, 10))

image = ax.imshow(
    matrix.values,
    cmap="YlOrRd",
    aspect="auto"
)

ax.set_xticks(range(len(matrix.columns)))
ax.set_xticklabels(matrix.columns, rotation=90)

ax.set_yticks(range(len(matrix.index)))
ax.set_yticklabels(matrix.index)

ax.set_title(
    "RoboOps | Telemetry Distribution Drift",
    fontsize=16,
    fontweight="bold"
)

fig.colorbar(
    image,
    ax=ax,
    label="Standardized Mean Shift"
)

plt.tight_layout()

plt.savefig(
    output / "data_drift_heatmap.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Drift visualization generated successfully.")
