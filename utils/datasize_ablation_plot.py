import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

params = {
    "legend.fontsize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 18,
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

_, axes = plt.subplots(1, 1, figsize = (8, 5))

axes.set_ylim(2.5, 20.0)
axes.set_yticks(np.arange(5, 22.5, 2.5))
axes.set_xlabel("Dataset size", fontsize = 18, labelpad = 12)
axes.set_ylabel("Top1 Accuracy (%) on ImageNet1K", fontsize = 18, labelpad = 12)
axes.set_xticks(np.arange(len(data["Metric"])))
axes.set_xticklabels(data["Metric"], fontsize = 18, rotation = 0)
axes.plot(np.arange(len(data["Metric"])), data["CLIP"], "^-", markersize = 12.5, label = "CLIP", color = "brown", alpha = 0.85)
axes.plot(np.arange(len(data["Metric"])), data["CyCLIP"], "*-", markersize = 15, label = "CyCLIP", alpha = 0.85)
axes.yaxis.grid()

axes.legend(bbox_to_anchor = (1.0, 1.05))
plt.tight_layout()

os.makedirs("analysis/plots", exist_ok = True)
plt.savefig(f"analysis/plots/datasize_ablation.png")