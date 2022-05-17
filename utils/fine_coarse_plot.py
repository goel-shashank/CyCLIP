import os
from cv2 import rotate
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
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [43.55,	34.18, 31.11, 25.11, 40.09, 54.49],
    "CyCLIP": [47.11, 35.46, 33.01, 26.42, 41.25, 55.36],
}

data_coarse_top_1 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [34.82, 52.11, 47.24, 37.03, 14.59, 40.51],
    "CyCLIP": [40.37, 56.79, 52.15, 41.57, 16.48, 44.43],
}

data_fine_top_2 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [67.65, 46.53, 43.85, 36.69, 57.93, 71.55],
    "CyCLIP": [68.67, 48.22, 45.37, 38.34, 57.89, 71.96],
}

data_coarse_top_2 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [49.51, 66.06, 61.42, 49.93, 24.71, 52.77],
    "CyCLIP": [56.26, 69.66, 65.27, 54.59, 26.75, 55.98],
}

data_fine_top_3 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [82.41, 53.95, 51.61, 44.66, 67.80, 79.52],
    "CyCLIP": [83.32, 55.43, 52.85, 46.09, 68.75, 79.81],
}

data_coarse_top_3 = {
    "Dataset": ["CIFAR100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [58.39, 73.20, 69.09, 57.61, 32.49, 60.07],
    "CyCLIP": [65.65, 76.45, 72.39, 62.21, 34.83, 63.12],
}

data = (eval(f"data_fine_top_{k}"), eval(f"data_coarse_top_{k}"))

figure, axes = plt.subplots(1, 2, figsize = (12, 5))

X_axis = np.arange(1, 7)
Y_axis = np.array([0, 10, 20, 30, 40, 50, 60, 70])

for index in range(2):
    if(index == 0):
        yaxis_label = f"Fine-grained Top{k} Accuracy (%)"
        xaxis_label = "(a)"
    else:
        yaxis_label = f"Coarse-grained Top{k} Accuracy (%)"
        xaxis_label = "(b)"

    axes[index].set_ylim(0, 80)
    axes[index].set_xlabel(xaxis_label)
    axes[index].set_ylabel(yaxis_label)
    axes[index].set_xticks(X_axis)
    axes[index].set_yticks(Y_axis)
    axes[index].set_xticklabels(data[index]["Dataset"], fontsize = 12, rotation = 30)
    axes[index].set_yticklabels(Y_axis, fontsize = 12, rotation = 0)
    axes[index].bar(X_axis, data[index]["CLIP"], label = "CLIP", width = 0.36, color = "brown")
    axes[index].bar(X_axis + 0.36, data[index]["CyCLIP"], width = 0.36, label = "CyCLIP")
    
    for i, v in enumerate(data[index]["CLIP"]):
        axes[index].text(i + 0.625, v + 1, v, fontsize = 11)
        axes[index].text(i + 1.225, data[index]["CyCLIP"][i] + 1, str(data[index]["CyCLIP"][i]), fontsize = 11)

    axes[index].legend(prop = {"size": 12})
    
plt.tight_layout()
plt.subplots_adjust(wspace = 0.225)

os.makedirs("analysis/plots", exist_ok = True)
plt.savefig(f"analysis/plots/fine_coarse_top{k}.png")