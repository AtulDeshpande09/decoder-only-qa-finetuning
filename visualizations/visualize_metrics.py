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
# 2. Style for research paper
# -----------------------------
sns.set(style="whitegrid", context="paper", font_scale=1.3)
palette = "Set2"

# -----------------------------
# 3. GROUPED BAR CHARTS
# One per dataset
# -----------------------------
metrics = ["BLEU", "ROUGE", "BERTScore"]

for dataset in df["Dataset"].unique():
    sub = df[df["Dataset"] == dataset]
    
    melted = sub.melt(id_vars=["Model"], value_vars=metrics,
                      var_name="Metric", value_name="Score")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette=palette)
    
    plt.title(f"Model Comparison on Dataset Size {dataset}")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"grouped_bar_{dataset}.png", dpi=300)
    plt.show()


# -----------------------------
# 4. LINE PLOTS (Scaling trend)
# One per metric
# -----------------------------
for metric in metrics:
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=df, x="Dataset", y=metric, hue="Model",
                 marker="o", palette="tab10")
    
    plt.title(f"{metric} vs Dataset Size")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"lineplot_{metric}.png", dpi=300)
    plt.show()


# -----------------------------
# 5. HEATMAPS
# Compact overview
# -----------------------------
for metric in metrics:
    pivot = df.pivot(index="Model", columns="Dataset", values=metric)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f")
    
    plt.title(f"Heatmap of {metric}")
    plt.tight_layout()
    plt.savefig(f"heatmap_{metric}.png", dpi=300)
    plt.show()