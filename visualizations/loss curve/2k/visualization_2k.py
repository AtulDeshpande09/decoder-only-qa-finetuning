import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Publication style
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
# Aligned loss data
# -----------------------------
loss_data = {
    "GPT-2": {
        "epoch": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "loss":  [4.0886, 3.5024, 4.1476, 3.8090, 4.0775, 3.2288, 2.8052]
    },
    "LLaMA-2-7B": {
        "epoch": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "loss":  [2.2064, 1.2520, 1.1746, 1.0605, 1.1085, 0.8918, 0.9832]
    },
    "Phi-2": {
        "epoch": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "loss":  [2.6090, 1.7299, 1.5328, 1.5185, 1.5004, 1.4156, 1.4808]
    },
    "Mistral": {
        "epoch": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "loss":  [1.8663, 1.1131, 0.7125, 0.6228, 0.9346, 0.4549, 0.5996]
    }
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6.5, 4.5))

line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "D", "^"]

for (model, values), ls, mk in zip(loss_data.items(), line_styles, markers):
    plt.plot(
        values["epoch"],
        values["loss"],
        linestyle=ls,
        marker=mk,
        linewidth=2,
        markersize=5,
        label=model
    )

# Labels
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison Across Transformer Models")

# Legend
plt.legend(frameon=True)

# Clean layout
plt.tight_layout()

# Save for paper
plt.savefig("loss_curve_new.pdf", bbox_inches="tight")
plt.savefig("loss_curve_new.png", dpi=300)

plt.show()
