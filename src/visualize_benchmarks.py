from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/model_stability.csv")

output = Path("docs/benchmarks")
output.mkdir(parents=True, exist_ok=True)

plt.style.use("ggplot")

metrics = [
    ("stability", "Model Stability", "Jaccard Similarity"),
    ("latency_ms", "Inference Latency", "Milliseconds"),
    ("model_size_kb", "Model Size", "Kilobytes"),
    ("training_seconds", "Training Time", "Seconds")
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, (column, title, ylabel) in zip(axes.flat, metrics):

    ax.bar(
        df["trees"].astype(str),
        df[column],
        color="#2563EB"
    )

    ax.set_title(title)
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel(ylabel)

fig.suptitle(
    "RoboOps | Model Reliability & Computational Performance",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    output / "model_benchmark.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# Cost-stability trade-off

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    df["latency_ms"],
    df["stability"],
    s=140,
    color="#059669"
)

for _, row in df.iterrows():
    ax.annotate(
        f"{int(row['trees'])} trees",
        (row["latency_ms"], row["stability"]),
        xytext=(8, 8),
        textcoords="offset points"
    )

ax.set_xlabel("Inference Latency (ms)")
ax.set_ylabel("Model Stability")
ax.set_title("RoboOps | Cost-Stability Trade-off")

plt.tight_layout()
plt.savefig(
    output / "cost_stability_tradeoff.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("Benchmark visualizations generated:", output)
