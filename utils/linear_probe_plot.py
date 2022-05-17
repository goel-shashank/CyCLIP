import os
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
    "Dataset": ["Caltech101", "CIFAR10", "CIFAR100", "DTD", "FGVCAircraft", "Flowers102", "Food101", "GTSRB", "ImageNet1K", "OxfordIIITPet", "RenderedSST2", "StanfordCars", "STL10", "SVHN"],
    "CLIP":    [76.80, 78.27, 72.37, 61.44, 28.32, 84.96, 54.47, 69.45, 35.47, 58.85, 53.10, 20.53, 89.76, 47.64],
    "CyCLIP":  [77.10, 77.34, 72.77, 64.47, 27.24, 84.72, 54.95, 71.70, 36.69, 58.10, 54.04, 22.72, 90.42, 48.16],
}

figure, axes = plt.subplots(3, 5, figsize = (15, 9))

for index in range(14):
    row, col = index // 5, index % 5
    axes[row][col].set_title(df["Dataset"][index])
    axes[row][col].set_ylim(0, 100)
    axes[row][0].set_ylabel("Top1 Accuracy (%)")
    axes[row][col].set_xticks([1.0, 1.25])
    axes[row][col].set_xticklabels(["CLIP", "CyCLIP"], fontsize = 14, rotation = 0)
    axes[row][col].bar(1.0, df["CLIP"][index], label = "CLIP", width = 0.15, color = "brown")
    axes[row][col].bar(1.25, df["CyCLIP"][index], label = "CyCLIP", width = 0.15)
    axes[row][col].text(1.0, df["CLIP"][index] + 1, str(df["CLIP"][index]), horizontalalignment = "center", fontsize = "x-large")
    axes[row][col].text(1.25, df["CyCLIP"][index] + 1, str(df["CyCLIP"][index]), horizontalalignment = "center", fontsize = "x-large")

figure.delaxes(axes[2][4])
figure.tight_layout()

os.makedirs("analysis/plots", exist_ok = True)
plt.savefig(f"analysis/plots/linear_probe_plot.png")