import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

params = {
    "axes.titlesize": 22,
    "legend.fontsize": 16,
    "figure.figsize": (12, 8),
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

data = {
    "Dataset": ["CIFAR10", "CIFAR100", "Food101", "OxfordIIITPet", "Flowers102", "StanfordCars", "DTD", "Caltech101", "FGCVAircraft"],
    "CLIP":   [78.33, 54.76, 54.58, 58.41, 83.67, 20.25, 59.95, 80.05, 27.96],
    "CyCLIP": [76.98, 55.46, 54.90, 57.86, 83.10, 22.40, 64.36, 80.74, 27.75],
}

figure, axes = plt.subplots(3, 3, figsize = (15, 5))

for index in range(9):
    row = index // 3
    col = index % 3

    X_axis = np.arange(1, 3)
    # Y_axis = np.array([0, 10, 20, 30, 40, 50, 60])

    # axes[index].set_ylim(0, 65)
    # axes[index].set_xlabel(xaxis_label)
    axes[index].set_ylabel("yaxis_label")
    axes[index].set_xticks(X_axis)
    # axes[index].set_yticks(Y_axis)
    axes[index].set_xticklabels(data[index]["Dataset"], fontsize = 12, rotation = 0)
    # axes[index].set_yticklabels(Y_axis, fontsize = 12, rotation = 0)
    axes[index].bar(X_axis, data[index]["CLIP"], label = "CLIP", width = 0.25, color = "brown")
    axes[index].bar(X_axis + 0.25, data[index]["CyCLIP"], width = 0.25, label = "CyCLIP")
    
    for i, v in enumerate(data[index]["CLIP"]):
        axes[index].text(i + 0.8, v + 1, v)
        axes[index].text(i + 1.1, data[index]["CyCLIP"][i] + 1, str(data[index]["CyCLIP"][i]))

    axes[index].legend(prop = {"size": 12})

os.makedirs("analysis/barplots", exist_ok = True)
plt.savefig(f"analysis/barplots/fine_coarse_top{k}.png")