from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/processed/training_sample.parquet")
OUTPUT = Path("data/processed/training_features.parquet")

WINDOW = 30

df = pd.read_parquet(INPUT)
df = df.sort_values(["source_file", "timestamp"])

sensor_columns = df.select_dtypes(include="number").columns

features = {}

for col in sensor_columns:

    group = df.groupby("source_file")[col]

    features[f"{col}_mean"] = group.transform(
        lambda s: s.rolling(WINDOW).mean()
    )

    features[f"{col}_std"] = group.transform(
        lambda s: s.rolling(WINDOW).std()
    )

    features[f"{col}_change"] = group.diff()

    features[f"{col}_rms"] = group.transform(
        lambda s: np.sqrt(s.pow(2).rolling(WINDOW).mean())
    )

    features[f"{col}_range"] = group.transform(
        lambda s: s.rolling(WINDOW).max()
        - s.rolling(WINDOW).min()
    )

feature_df = pd.DataFrame(features, index=df.index)

result = pd.concat([df, feature_df], axis=1)

result = result.replace([np.inf, -np.inf], np.nan)
result = result.dropna().reset_index(drop=True)

result.to_parquet(OUTPUT, index=False, compression="snappy")

print("\n--- TIME-SERIES FEATURE REPORT ---")
print("Original shape:", df.shape)
print("New features:", len(features))
print("Final shape:", result.shape)
print("Missing values:", result.isna().sum().sum())
print("Memory usage (MB):",
      round(result.memory_usage(deep=True).sum() / 1024**2, 2))
print("Output:", OUTPUT)from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/processed/training_sample.parquet")
OUTPUT_PATH = Path("data/processed/training_features.parquet")

WINDOW = 30

df = pd.read_parquet(INPUT_PATH)

df = df.sort_values(["source_file", "timestamp"])

sensor_columns = df.select_dtypes(include="number").columns

features = {}

for column in sensor_columns:

    grouped = df.groupby("source_file")[column]

    features[f"{column}_rolling_mean"] = grouped.transform(
        lambda x: x.rolling(WINDOW).mean()
    )

    features[f"{column}_rolling_std"] = grouped.transform(
        lambda x: x.rolling(WINDOW).std()
    )

    features[f"{column}_change"] = grouped.diff()

    features[f"{column}_rms"] = grouped.transform(
        lambda x: np.sqrt(x.pow(2).rolling(WINDOW).mean())
    )

    features[f"{column}_range"] = grouped.transform(
        lambda x: x.rolling(WINDOW).max()
        - x.rolling(WINDOW).min()
    )

feature_df = pd.DataFrame(features, index=df.index)

result = pd.concat([df, feature_df], axis=1)

result = result.replace([np.inf, -np.inf], np.nan)
result = result.dropna().reset_index(drop=True)

result.to_parquet(
    OUTPUT_PATH,
    index=False,
    compression="snappy"
)

print("\n--- FEATURE ENGINEERING REPORT ---")
print("Original shape:", df.shape)
print("Engineered features:", len(features))
print("Final shape:", result.shape)
print("Missing values:", result.isna().sum().sum())
print("Output:", OUTPUT_PATH)
