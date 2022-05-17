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
    "CLIP": [5.03, 11.1, 15.10, 17.28],
    "CyCLIP": [6.19, 12.12, 16.57, 18.68],
}

_, axes = plt.subplots(1, 1, figsize = (8, 6))

axes.set_ylim(0, 25)
axes.set_yticks(np.arange(0, 25, 2.5))
axes.set_xlabel("Dataset size", fontsize = 18, labelpad = 10)
axes.set_ylabel("Zero-shot Accuracy (%) on ImageNet1K", fontsize = 18, labelpad = 12)
axes.set_xticks(np.arange(len(data["Metric"])))
axes.set_xticklabels(data["Metric"], fontsize = 18, rotation = 0)
axes.plot(np.arange(len(data["Metric"])), data["CLIP"], "o-", markersize = 8, label = "CLIP", color = "brown")
axes.plot(np.arange(len(data["Metric"])), data["CyCLIP"], "D-", markersize = 8, label = "CyCLIP")
axes.legend()
axes.yaxis.grid()

plt.tight_layout()

os.makedirs("analysis/plots", exist_ok = True)
plt.savefig(f"analysis/plots/datasize_ablation.png")