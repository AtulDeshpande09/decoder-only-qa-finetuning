import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Create DataFrame
# -----------------------------
data = {
    "Dataset": [200]*4 + [1000]*4 + [2306]*4,
    "Model": ["GPT-2", "Mistral", "LLaMA-2", "Phi-2"] * 3,
    
    "BLEU": [
        0.0339, 0.1372, 0.1094, 0.0686,
        0.0274, 0.1700, 0.1339, 0.1431,
        0.0204, 0.1573, 0.1278, 0.1215
    ],
    
    "ROUGE": [
        0.1669, 0.3853, 0.3232, 0.3096,
        0.1752, 0.3904, 0.3343, 0.3396,
        0.1717, 0.3529, 0.3287, 0.3320
    ],
    
    "BERTScore": [
        0.8803, 0.9228, 0.9131, 0.9120,
        0.8829, 0.9196, 0.9126, 0.9125,
        0.8808, 0.9126, 0.9119, 0.9109
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Style (IEEE friendly)
# -----------------------------
sns.set(style="whitegrid", context="paper", font_scale=1.2)

# -----------------------------
# FIGURE 1: Scaling Trends
# 3-panel line plot
# -----------------------------
metrics = ["BLEU", "ROUGE", "BERTScore"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)

for i, metric in enumerate(metrics):
    sns.lineplot(
        data=df,
        x="Dataset",
        y=metric,
        hue="Model",
        marker="o",
        ax=axes[i]
    )
    
    axes[i].set_title(metric)
    axes[i].set_ylim(0, 1)
    axes[i].set_xlabel("Dataset Size")
    
    if i != 0:
        axes[i].set_ylabel("")
        axes[i].legend_.remove()

# Keep legend only once
axes[0].legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig("figure1_scaling_trends.png", dpi=300)
plt.show()


# -----------------------------
# FIGURE 2: Grouped Bars
# Largest dataset comparison
# -----------------------------
largest = df[df["Dataset"] == 2306]

melted = largest.melt(
    id_vars=["Model"],
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(7, 5))
sns.barplot(
    data=melted,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Model Comparison on Largest Dataset")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("figure2_grouped_bar.png", dpi=300)
plt.show()