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
        "epoch": [0.0, 	0.5, 	1.0, 	1.5, 	2.0, 	2.5, 	3.0],
        "loss":  [3.8825, 4.2773, 3.0549, 3.9294, 3.1193, 2.2465, 2.4737]
    },
    "LLaMA-2-7B": {
        "epoch": [0.0, 	0.5, 	1.0, 	1.5, 	2.0, 	2.5, 	3.0],
        "loss":  [2.1101, 1.3101, 1.1707, 1.1139, 1.1021, 1.0366, 1.0843]
    },
    "Phi-2": {
        "epoch": [0.0, 	0.5, 	1.0, 	1.5, 	2.0, 	2.5, 	3.0],
        "loss":  [2.4239, 1.7293, 1.6082, 1.5485, 1.5997, 1.5835, 1.5853]
    },
    "Mistral": {
        "epoch": [0.0, 	0.5, 	1.0, 	1.5, 	2.0, 	2.5, 	3.0],
        "loss":  [1.7420, 1.1260, 0.6469, 0.6522, 0.8343, 0.6243, 0.5984]
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
