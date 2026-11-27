import pickle
from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import jaccard_score

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

    predictions = []
    training_times = []
    latencies = []
    model_sizes = []

    for seed in [42, 100, 202]:

        model = IsolationForest(
            n_estimators=trees,
            contamination=0.01,
            random_state=seed,
            n_jobs=2
        )

        start = perf_counter()
        model.fit(X_train)
        training_times.append(perf_counter() - start)

        start = perf_counter()
        flags = model.predict(X_val) == -1
        elapsed = perf_counter() - start

        predictions.append(flags)

        latencies.append(
            elapsed / len(X_val) * 1000
        )

        model_sizes.append(
            len(pickle.dumps(model)) / 1024
        )

    agreements = [
        jaccard_score(a, b, zero_division=0)
        for a, b in combinations(predictions, 2)
    ]

    results.append({
        "trees": trees,
        "stability": np.mean(agreements),
        "training_seconds": np.mean(training_times),
        "latency_ms": np.mean(latencies),
        "model_size_kb": np.mean(model_sizes),
        "average_flags": np.mean([
            p.sum() for p in predictions
        ])
    })

report = pd.DataFrame(results)

print("\n--- MODEL STABILITY BENCHMARK ---")
print(report.round(4).to_string(index=False))

report.to_csv(
    "data/processed/model_stability.csv",
    index=False
)
