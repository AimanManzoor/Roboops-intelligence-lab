from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/training_sample.parquet")
OUTPUT_DIR = Path("data/processed/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
df = df.sort_values("timestamp")

print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicate timestamps:", df["timestamp"].duplicated().sum())
print("Start time:", df["timestamp"].min())
print("End time:", df["timestamp"].max())

print("\n--- SAMPLING INTERVAL ---")
intervals = df["timestamp"].diff().dt.total_seconds()
print(intervals.describe())

print("\n--- SENSOR STATISTICS ---")
sensor_columns = df.select_dtypes(include="number").columns
print(df[sensor_columns].describe().round(3))

df[sensor_columns].describe().to_csv(
    OUTPUT_DIR / "sensor_statistics.csv"
)

# Sensor distributions
for measurement in ["speed", "torque", "angle"]:

    columns = [
        f"axis_{axis}_{measurement}"
        for axis in range(1, 8)
    ]

    df[columns].hist(
        bins=40,
        figsize=(14, 10)
    )

    plt.suptitle(f"Robot {measurement.title()} Distributions")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{measurement}_distribution.png",
        dpi=150
    )

    plt.close()

# Sensor correlation
correlation = df[sensor_columns].corr()

fig, ax = plt.subplots(figsize=(14, 12))

image = ax.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(sensor_columns)))
ax.set_yticks(range(len(sensor_columns)))

ax.set_xticklabels(sensor_columns, rotation=90)
ax.set_yticklabels(sensor_columns)

fig.colorbar(image, ax=ax)
ax.set_title("Robot Sensor Correlation")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150)
plt.close()

print("\nEDA completed successfully.")
print("Results saved to:", OUTPUT_DIR)
