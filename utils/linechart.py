import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

params = {
    "legend.fontsize": 18,
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "xtick.labelsize": 22,
    "ytick.labelsize": 18,
    "figure.titlesize": 24,
    "font.family": "Liberation Mono"
}

plt.rcParams.update(params)
plt.style.use("seaborn-whitegrid")
sns.set_style("white")

data = {
    "Metric": ["500K", "1M", "2M", "3M"],
    "CLIP": [5.03, 11.1, 15.10, 20.03],
    "CyCLIP": [6.19, 12.12, 16.57, 22.08],
}

_, axes = plt.subplots(1, 1, figsize = (8, 6))

axes.set_xlabel("ImageNet1K", fontsize = 18, labelpad = 10)
axes.set_ylabel("Top-1 Zero-shot Accuracy (%)", fontsize = 18, labelpad = 12)
axes.set_xticks(np.arange(4))
axes.set_xticklabels(data["Metric"], fontsize = 18, rotation = 0)
axes.plot(np.arange(4), data["CLIP"], "o-", markersize = 8, label = "CLIP", color = "brown")
axes.plot(np.arange(4), data["CyCLIP"], "D-", markersize = 8, label = "CyCLIP")
axes.legend()

plt.tight_layout()
plt.show()

os.makedirs("analysis/linecharts", exist_ok = True)
plt.savefig(f"analysis/linecharts/datasize_ablation.png")