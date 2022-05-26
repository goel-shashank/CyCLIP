import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

params = {
    "figure.figsize": (8, 6),
    "axes.titlesize": 22,
    "legend.fontsize": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.titlesize": 22,
    "font.family": "Liberation Mono"
}

plt.rcParams.update(params)
plt.style.use("seaborn-whitegrid")
sns.set_style("white")


ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer = True))

file = "data/ImageNet-A/classes.py"
superclasses = eval(open(file).read())["superclasses"]

data = list(map(len, superclasses))

_, bins, patches = plt.hist(data, 25, color = "green", edgecolor = "black", linewidth = 0.75)
centers = 0.5 * (bins[:-1] + bins[1:])
colors = centers - min(centers)
colors /= max(colors)

colormap = plt.cm.get_cmap("RdYlBu_r")
for color, patch in zip(colors, patches):
    plt.setp(patch, "facecolor", colormap(color))

plt.xlabel("Number of subclasses per superclass", labelpad = 12)
plt.tight_layout()

os.makedirs("analysis/plots", exist_ok = True)
plt.savefig(f"analysis/plots/superclass_distribution.ImageNet-A.png")