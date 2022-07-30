import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

params = {
    "figure.figsize": (10, 5),
    "legend.fontsize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.family": "Liberation Mono"
}

plt.rcParams.update(params)
plt.style.use("seaborn-whitegrid")
sns.set_style("white")

clip = {
    "Datasize": ["500K", "1M", "2M", "3M"],
    "ImageNet1K": [5.03, 11.1, 15.07, 17.28],
    "ImageNetV2": [4.53, 9.33, 12.65, 14.39]
}

cyclip = {
    "Datasize": ["500K", "1M", "2M", "3M"],
    "ImageNet1K": [6.19, 12.12, 16.57, 18.68],
    "ImageNetV2": [5.33, 9.88, 13.89, 15.31]
}

plt.clf()
    
plt.scatter(clip["ImageNet1K"][0], clip["ImageNetV2"][0], marker = "^", s = 80, label = f"CLIP-{clip['Datasize'][0]}")
plt.scatter(clip["ImageNet1K"][1], clip["ImageNetV2"][1], marker = "^", s = 80, label = f"CLIP-{clip['Datasize'][1]}")
plt.scatter(clip["ImageNet1K"][2], clip["ImageNetV2"][2], marker = "^", s = 80, label = f"CLIP-{clip['Datasize'][2]}")
plt.scatter(clip["ImageNet1K"][3], clip["ImageNetV2"][3], marker = "^", s = 80, label = f"CLIP-{clip['Datasize'][3]}")

plt.scatter(cyclip["ImageNet1K"][0], cyclip["ImageNetV2"][0], marker = "*", s = 96, label = f"CyCLIP-{cyclip['Datasize'][0]}")
plt.scatter(cyclip["ImageNet1K"][1], cyclip["ImageNetV2"][1], marker = "*", s = 96, label = f"CyCLIP-{cyclip['Datasize'][1]}")
plt.scatter(cyclip["ImageNet1K"][2], cyclip["ImageNetV2"][2], marker = "*", s = 96, label = f"CyCLIP-{cyclip['Datasize'][2]}")
plt.scatter(cyclip["ImageNet1K"][3], cyclip["ImageNetV2"][3], marker = "*", s = 96, label = f"CyCLIP-{cyclip['Datasize'][3]}")

plt.xlabel("Top1 Accuracy on ImageNet1K (%)", labelpad = 12)
plt.ylabel(f"Top1 Accuracy on ImageNetV2 (%)", labelpad = 12)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle = "--", color = "k", lw = 2, scalex = False, scaley = False, label = "y = x")

ypoints = plt.ylim()
xpoints = [(y + 9) / 1.2 for y in ypoints]
plt.plot(xpoints, ypoints, linestyle = "-", color = "r", lw = 1, label = "Linear fit to\nstandard training")

plt.yticks(np.arange(5.0, 22.5, 2.5))
plt.xticks(np.arange(5.0, 25.0, 2.5))
plt.ylim(2.5, 20.0)
plt.xlim(2.5, 22.5)
    
plt.legend(bbox_to_anchor = (1.0, 1.05))
plt.grid()
plt.tight_layout()

os.makedirs("analysis/plots", exist_ok = True)
plt.savefig(f"analysis/plots/effective_robustness_plot.png")