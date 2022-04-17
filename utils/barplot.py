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
    "Dataset": ["CIFAR100", "ImageNet", "ImageNetV2", "ImageNetSketch"],
    "CLIP": [43.55,	34.18, 31.11, 25.11],
    "CyCLIP": [47.11, 35.46, 33.01, 26.42],
}

data_coarse_top_1 = {
    "Dataset": ["CIFAR100", "ImageNet", "ImageNetV2", "ImageNetSketch"],
    "CLIP": [34.82, 52.10, 47.24, 37.03],
    "CyCLIP": [40.37, 56.79, 52.15, 41.57],
}

data_fine_top_2 = {
    "Dataset": ["CIFAR100", "ImageNet", "ImageNetV2", "ImageNetSketch"],
    "CLIP": [67.65, 46.53, 43.85, 36.69],
    "CyCLIP": [68.67, 48.22, 45.37, 38.34],
}

data_coarse_top_2 = {
    "Dataset": ["CIFAR100", "ImageNet", "ImageNetV2", "ImageNetSketch"],
    "CLIP": [49.51, 66.06, 61.42, 49.93],
    "CyCLIP": [56.26, 69.66, 65.27, 54.59],
}

data_fine_top_3 = {
    "Dataset": ["CIFAR100", "ImageNet", "ImageNetV2", "ImageNetSketch"],
    "CLIP": [82.41, 53.95, 51.61, 44.66],
    "CyCLIP": [83.32, 55.43, 52.85, 46.09],
}

data_coarse_top_3 = {
    "Dataset": ["CIFAR100", "ImageNet", "ImageNetV2", "ImageNetSketch"],
    "CLIP": [58.39, 73.20, 69.09, 57.61],
    "CyCLIP": [65.65, 76.45, 72.39, 62.21],
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