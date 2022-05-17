import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

params = {
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 14,
    "font.family": "Liberation Mono"
}

plt.rcParams.update(params)
plt.style.use("seaborn-whitegrid")
sns.set_style("white")

data = [None, None, None]

data[0] = ("(a) CIFAR10", {
    "CLIP": [44.60, 46.04, 47.06, 48.45],
    "CyCLIP": [48.81, 50.89, 52.30, 53.71],
})

data[1] = ("(b) CIFAR100", {
    "CLIP": [16.21, 17.28, 18.42, 19.36],
    "CyCLIP": [20.43, 21.96, 23.18, 24.31],
})

data[2] = ("(c) ImageNet1K", {
    "CLIP": [16.34, 17.42, 18.58, 19.78],
    "CyCLIP": [19.20, 20.31, 21.95, 23.94],
})

figure, axes = plt.subplots(1, 3, figsize = (16, 5))

for index in range(3):
    axes[index].set_xlabel(data[index][0], fontsize = 16, labelpad = 12)
    if(index == 0): axes[index].set_ylabel("Consistency Score (%)", labelpad = 12)
    axes[index].set_xticks(np.arange(4))
    axes[index].set_xticklabels(["Top-1", "Top-3", "Top-5", "Top-10"], fontsize = 12, rotation = 0)
    axes[index].plot(np.arange(4), data[index][1]["CLIP"], "*-", markersize = 15, label = "CLIP", color = "brown")
    axes[index].plot(np.arange(4), data[index][1]["CyCLIP"], "^-", markersize = 15, label = "CyCLIP")
    axes[index].legend(prop = {"size": 14})

plt.tight_layout()
plt.subplots_adjust(wspace = 0.225)

os.makedirs("analysis/plots", exist_ok = True)
plt.savefig(f"analysis/plots/consistency.png")