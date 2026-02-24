import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Style (publication quality)
# -----------------------------
sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

# -----------------------------
# Loss data
# -----------------------------
loss_data = {
    "GPT-2": {
        "epoch": [0.01, 1.0, 2.0, 3.0],
        "loss": [4.22, 3.38, 2.92, 2.68]
    },
    "LLaMA-2-7B": {
        "epoch": [0.44, 0.89, 1.31, 1.76, 2.18, 2.62],
        "loss": [2.07, 1.60, 1.39, 1.26, 1.25, 1.16]
    },
    "Phi-2": {
        "epoch": [0.44, 0.89, 1.31, 1.76, 2.18, 2.62],
        "loss": [2.34, 2.03, 1.86, 1.67, 1.60, 1.55]
    },
    "Mistral": {
        "epoch": [0.04, 1.0, 2.0, 3.0],
        "loss": [1.62, 1.45, 0.92, 1.03]
    }
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6.5, 4.5))

markers = ["o", "s", "D", "^"]

for (model, values), marker in zip(loss_data.items(), markers):
    plt.plot(
        values["epoch"],
        values["loss"],
        marker=marker,
        linewidth=2,
        markersize=5,
        label=model
    )

# Labels and title
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison Across Transformer Models")

# Legend
plt.legend(frameon=True)

# Layout
plt.tight_layout()

# Save figure (important for papers)
plt.savefig("loss_curve.pdf", bbox_inches="tight")
plt.savefig("loss_curve.png", dpi=300)

plt.show()
