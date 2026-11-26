from time import perf_counter

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold

df = pd.read_parquet(
    "data/processed/training_features.parquet"
).sort_values("timestamp")

X = df.select_dtypes(include="number")

split = int(len(X) * 0.8)

selector = VarianceThreshold()

X_train = selector.fit_transform(X.iloc[:split])
X_val = selector.transform(X.iloc[split + 29:])

results = []

for trees in [25, 50, 100, 200]:

    model = IsolationForest(
        n_estimators=trees,
        contamination=0.01,
        random_state=42,
        n_jobs=2
    )

    start = perf_counter()
    model.fit(X_train)
    training_time = perf_counter() - start

    start = perf_counter()
    predictions = model.predict(X_val)
    inference_time = perf_counter() - start

    results.append({
        "trees": trees,
        "training_seconds": training_time,
        "inference_seconds": inference_time,
        "latency_ms": inference_time / len(X_val) * 1000,
        "flagged_records": int((predictions == -1).sum())
    })

report = pd.DataFrame(results)

print("\n--- COST-PERFORMANCE BENCHMARK ---")
print(report.round(4).to_string(index=False))

report.to_csv(
    "data/processed/model_benchmark.csv",
    index=False
)
