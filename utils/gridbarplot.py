import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

params = {
    "axes.titlesize": 22,
    "legend.fontsize": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.titlesize": 22,
    "font.family": "Liberation Mono"
}

plt.rcParams.update(params)
plt.style.use("seaborn-whitegrid")
sns.set_style("white")

df = {
    "Dataset": ["CIFAR10", "CIFAR100", "FOOD", "PETS", "FLOWERS", "CARS", "DTD", "CALTECH", "AIRCRAFT"],
    "CLIP":   [78.33, 54.76, 54.58, 58.41, 83.67, 20.25, 59.95, 80.05, 27.96],
    "CyCLIP": [76.98, 55.46, 54.90, 57.86, 83.10, 22.40, 64.36, 80.74, 27.75],
}

figure, axes = plt.subplots(3, 3, figsize = (9, 9))

for index in range(9):
    row, col = index // 3, index % 3
    axes[row][col].set_title(df["Dataset"][index])
    axes[row][col].set_ylim(0, 100)
    axes[row][0].set_ylabel("Top1 Accuracy (%)")
    axes[row][col].set_xticks([1.0, 1.25])
    axes[row][col].set_xticklabels(["CLIP", "CyCLIP"], fontsize = 14, rotation = 0)
    axes[row][col].bar(1.0, df["CLIP"][index], label = "CLIP", width = 0.15, color = "brown")
    axes[row][col].bar(1.25, df["CyCLIP"][index], label = "CyCLIP", width = 0.15)
    axes[row][col].text(1.0, df["CLIP"][index] + 1, str(df["CLIP"][index]), horizontalalignment = "center", fontsize = "x-large")
    axes[row][col].text(1.25, df["CyCLIP"][index] + 1, str(df["CyCLIP"][index]), horizontalalignment = "center", fontsize = "x-large")

figure.tight_layout()

os.makedirs("analysis/barplots", exist_ok = True)
plt.savefig(f"analysis/barplots/gridbarplot.png")