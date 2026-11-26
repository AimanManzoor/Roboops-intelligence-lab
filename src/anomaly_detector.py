from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold

df = pd.read_parquet(
    "data/processed/training_features.parquet"
).sort_values("timestamp")

X = df.select_dtypes(include="number")

split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_val = X.iloc[split + 29:]

# Remove constant features
selector = VarianceThreshold()
X_train_selected = selector.fit_transform(X_train)
X_val_selected = selector.transform(X_val)

model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42,
    n_jobs=2
)

start = perf_counter()
model.fit(X_train_selected)
training_time = perf_counter() - start

start = perf_counter()
predictions = model.predict(X_val_selected)
inference_time = perf_counter() - start

anomalies = int((predictions == -1).sum())

print("\n--- BASELINE MODEL REPORT ---")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Original features:", X.shape[1])
print("Selected features:", X_train_selected.shape[1])
print("Flagged observations:", anomalies)
print("Flagged percentage:", round(anomalies / len(X_val) * 100, 2))
print("Training time (seconds):", round(training_time, 4))
print("Inference time (seconds):", round(inference_time, 4))
print("Inference latency (ms/record):",
      round(inference_time / len(X_val) * 1000, 4))
