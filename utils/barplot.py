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

k = 1

data_fine_top_1 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetSketch", "ImageNetV2"],
    "CLIP": [43.55,	34.18, 25.11, 31.11],
    "CyCLIP": [47.11, 35.46, 26.42, 33.01],
}

data_coarse_top_1 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetSketch", "ImageNetV2"],
    "CLIP": [34.82, 52.10, 37.03, 47.24],
    "CyCLIP": [40.37, 56.79, 41.57, 52.15],
}

data_fine_top_2 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetSketch", "ImageNetV2"],
    "CLIP": [67.65, 46.53, 36.69, 43.85],
    "CyCLIP": [68.67, 48.22, 38.34, 45.37],
}

data_coarse_top_2 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetSketch", "ImageNetV2"],
    "CLIP": [49.51, 66.06, 49.93, 61.42],
    "CyCLIP": [56.26, 69.66, 54.59, 65.27],
}

data_fine_top_3 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetSketch", "ImageNetV2"],
    "CLIP": [82.41, 53.95, 44.66, 51.61],
    "CyCLIP": [83.32, 55.43, 46.09, 52.85],
}

data_coarse_top_3 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetSketch", "ImageNetV2"],
    "CLIP": [58.39, 73.20, 57.61, 69.09],
    "CyCLIP": [65.65, 76.45, 62.21, 72.39],
}

data = (eval(f"data_fine_top_{k}"), eval(f"data_coarse_top_{k}"))

figure, axes = plt.subplots(1, 2, figsize = (15, 5))

X_axis = np.arange(1, 5)
Y_axis = np.array([0, 10, 20, 30, 40, 50, 60])

for index in range(2):
    if(index == 0):
        yaxis_label = "Fine-grained Top1 Accuracy (%)"
        xaxis_label = "(a)"
    else:
        yaxis_label = "Coarse-grained Top1 Accuracy (%)"
        xaxis_label = "(b)"

    axes[index].set_ylim(0, 65)
    axes[index].set_xlabel(xaxis_label)
    axes[index].set_ylabel(yaxis_label)
    axes[index].set_xticks(X_axis)
    axes[index].set_yticks(Y_axis)
    axes[index].set_xticklabels(data[index]["Dataset"], fontsize = 12, rotation = 0)
    axes[index].set_yticklabels(Y_axis, fontsize = 12, rotation = 0)
    axes[index].bar(X_axis, data[index]["CLIP"], label = "CLIP", width = 0.25, color = "brown")
    axes[index].bar(X_axis + 0.25, data[index]["CyCLIP"], width = 0.25, label = "CyCLIP")
    
    for i, v in enumerate(data[index]["CLIP"]):
        axes[index].text(i + 0.8, v + 1, v)
        axes[index].text(i + 1.1, data[index]["CyCLIP"][i] + 1, str(data[index]["CyCLIP"][i]))

    axes[index].legend(prop = {"size": 12})

os.makedirs("analysis/barplots", exist_ok = True)
plt.savefig(f"analysis/barplots/fine_coarse_top{k}.png")