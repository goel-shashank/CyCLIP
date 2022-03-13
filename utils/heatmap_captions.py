import os
import torch
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

def heatmap(options):
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pickle.load(open(options.input, "rb"))
    image_embeddings, text_embeddings = torch.tensor(data["image_embeddings"]).to(options.device), torch.tensor(data["text_embeddings"]).to(options.device)

    dataset = torch.utils.data.TensorDataset(image_embeddings, text_embeddings)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, shuffle = False, drop_last = True)
    
    heatmap = torch.zeros((options.batch_size, options.batch_size)).to(options.device)
    
    with torch.no_grad():
        for image_embeddings, text_embeddings in tqdm(dataloader):
            image_embeddings, text_embeddings = image_embeddings.to(options.device), text_embeddings.to(options.device)
            logits = image_embeddings @ text_embeddings.t()
            heatmap += torch.abs(logits - logits.t())
        
        heatmap /= len(dataloader)
        heatmap = heatmap.detach().cpu().numpy()
    
    print(heatmap)
    
    # vcenter, vmax = 0.0865, 0.0895
    # # vcenter, vmax = 0.0152, 0.0162
    
    # ax = sns.heatmap(
    #     heatmap, 
    #     cmap = sns.light_palette("#008000", n_colors = 25),
    #     square = True,
    #     xticklabels = False,
    #     yticklabels = False,
    #     norm = TwoSlopeNorm(vmin = 0, vcenter = vcenter, vmax = vmax)
    # )
    
    # ax.collections[0].colorbar.set_ticks([0, vcenter, vmax])
    # ax.collections[0].colorbar.set_ticklabels([0, vcenter, vmax])
    
    # os.makedirs(options.output, exist_ok = True)
    # plt.savefig(os.path.join(options.output, "diff.png")) 
    
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()

    parser.add_argument("-i,--input", dest = "input", type = str, default = "analysis/embeddings/baseline/CC3M.validation.pkl", help = "Input file")
    parser.add_argument("-o,--output", dest = "output", type = str, default = "analysis/heatmaps/baseline/CC3M/validation/", help = "Output directory")
    parser.add_argument("-b,--batch_size", dest = "batch_size", type = str, default = 8, help = "Grid/Batch Size")
    
    options = parser.parse_args()
    heatmap(options)