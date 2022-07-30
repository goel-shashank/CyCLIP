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

fine_top_1 = {
    "Dataset": ["CIFAR-100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [43.55,	34.18, 31.10, 25.11, 40.09, 54.49],
    "CyCLIP": [47.11, 35.46, 33.01, 26.42, 41.25, 55.36],    
    "C-CyCLIP": [49.19, 35.12, 32.31, 26.37, 42.56, 55.59],
    "I-CyCLIP": [47.71, 34.92, 32.21, 24.89, 39.60, 53.99]
}

coarse_top_1 = {
    "Dataset": ["CIFAR-100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [34.82, 52.11, 47.24, 37.03, 14.59, 40.51],
    "CyCLIP": [40.37, 56.79, 52.15, 41.57, 16.48, 44.43],    
    "C-CyCLIP": [43.06, 56.08, 50.89, 43.02, 16.40, 45.41],
    "I-CyCLIP": [39.30, 55.78, 51.48, 39.18, 15.63, 41.63]
}

fine_top_2 = {
    "Dataset": ["CIFAR-100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [67.65, 46.53, 43.85, 36.69, 57.93, 71.55],
    "CyCLIP": [68.67, 48.22, 45.37, 38.34, 57.89, 71.96],    
    "C-CyCLIP": [71.02, 47.95, 45.02, 37.67, 59.41, 72.06],
    "I-CyCLIP": [69.30, 47.42, 45.00, 37.24, 56.67, 70.99]
}

coarse_top_2 = {
    "Dataset": ["CIFAR-100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [49.51, 66.06, 61.42, 49.93, 24.71, 52.77],
    "CyCLIP": [56.26, 69.67, 65.27, 54.59, 26.75, 55.98],    
    "C-CyCLIP": [58.62, 69.74, 65.23, 56.41, 26.76, 57.21],
    "I-CyCLIP": [54.20, 68.46, 63.85, 51.58, 25.72, 51.94]
}

fine_top_3 = {
    "Dataset": ["CIFAR-100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [82.41, 53.95, 51.61, 44.66, 67.80, 79.52],
    "CyCLIP": [83.32, 55.43, 52.85, 46.09, 68.75, 79.81],    
    "C-CyCLIP": [85.26, 55.14, 52.40, 45.41, 69.99, 79.60],
    "I-CyCLIP": [84.07, 54.95, 52.03, 45.30, 67.92, 79.41]
}

coarse_top_3 = {
    "Dataset": ["CIFAR-100", "ImageNet1K", "ImageNetV2", "ImageNetSketch", "ImageNet-A", "ImageNet-R"],
    "CLIP": [58.39, 73.20, 69.09, 57.61, 32.49, 60.07],
    "CyCLIP": [65.65, 76.45, 72.40, 62.22, 34.83, 63.12],    
    "C-CyCLIP": [67.91, 76.76, 72.55, 64.11, 35.25, 64.3],
    "I-CyCLIP": [63.30, 74.98, 70.44, 58.68, 32.76, 58.48]
}

k = 1
data = (eval(f"fine_top_{k}"), eval(f"coarse_top_{k}"))

figure, axes = plt.subplots(2, 1, figsize = (14, 10))

X_axis = np.arange(1, 7)
Y_axis = np.arange(0, 70, 10)

for index in range(2):
    axes[index].set_ylim(0, 70)
    axes[index].set_xlabel(f"{'(a) Fine-grained' if (index == 0) else '(b) Coarse-grained'}", labelpad = 14)
    axes[index].set_ylabel(f"Top{k} Accuracy (%)", labelpad = 12)
    axes[index].set_xticks(X_axis)
    axes[index].set_yticks(Y_axis)
    axes[index].set_xticklabels(data[index]["Dataset"], fontsize = 14)
    axes[index].set_yticklabels(Y_axis, fontsize = 14)
    axes[index].bar(X_axis - 1.5 * 0.20, data[index]["CLIP"], width = 0.185, label = "CLIP", alpha = 0.6, color = "brown")
    axes[index].bar(X_axis - 0.5 * 0.20, data[index]["CyCLIP"], width = 0.185, label = "CyCLIP", alpha = 0.6)
    axes[index].bar(X_axis + 0.5 * 0.20, data[index]["C-CyCLIP"], width = 0.185, label = "C-CyCLIP", alpha = 0.6)
    axes[index].bar(X_axis + 1.5 * 0.20, data[index]["I-CyCLIP"], width = 0.185, label = "I-CyCLIP", alpha = 0.6)
    
    for i in range(len(data[index]["Dataset"])):
        for j, model in enumerate(["CyCLIP", "C-CyCLIP", "I-CyCLIP"]):
            text = ("+" if (data[index][model][i] > data[index]["CLIP"][i]) else "-") + str(round(abs(data[index][model][i] - data[index]["CLIP"][i]), 1))
            axes[index].text(i + 0.73 + j * 0.24, data[index][model][i] + 1.5, text, fontsize = 12)

    axes[index].legend(prop = {"size": 14}, bbox_to_anchor = (1.175, 1.025))
    
plt.tight_layout()
plt.subplots_adjust(hspace = 0.25)

os.makedirs("analysis/plots", exist_ok = True)
plt.savefig(f"analysis/plots/fine_coarse_top{k}.supplementary.png")