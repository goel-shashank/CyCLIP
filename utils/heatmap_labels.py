import os
import torch
import pickle
import warnings
import argparse
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

def heatmap(options):
    with torch.no_grad():
        options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = pickle.load(open(options.input, "rb"))
        image_embeddings, text_embeddings, labels, classes = torch.tensor(data["image_embeddings"]), torch.tensor(data["text_embeddings"]), torch.tensor(data["labels"]), data["classes"]
        
        if(len(options.classes) == 0):
            options.classes = classes
        
        indices_classes = [classes.index(c) for c in options.classes]
        indices_examples = list(itertools.chain(*zip(*[torch.arange(len(labels))[labels == classes.index(c)].tolist() for c in options.classes])))

        image_embeddings = image_embeddings[indices_examples, ...]
        text_embeddings = text_embeddings[indices_classes, ...].to(options.device)
        
        batch_size = len(options.classes)
        
        dataset = torch.utils.data.TensorDataset(image_embeddings)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, drop_last = True)
        
        heatmap = torch.zeros((batch_size, batch_size)).to(options.device)
        
        for batch in tqdm(dataloader):
            image_embeddings = batch[0]
            image_embeddings = image_embeddings.to(options.device)
            heatmap += image_embeddings @ text_embeddings.t()
        
        heatmap /= len(dataloader)
        heatmap = heatmap.detach().cpu().numpy()
        
        print(heatmap)
        
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()

    parser.add_argument("-i,--input", dest = "input", type = str, default = "analysis/embeddings/symmetric/CIFAR10.pkl", help = "Input file")
    parser.add_argument("-o,--output", dest = "output", type = str, default = "analysis/heatmaps/symmetric/CIFAR10/", help = "Output directory")
    parser.add_argument("-c,--classes", dest = "classes", nargs = "+", default = [], help = "Classes in the Grid")
    
    options = parser.parse_args()
    heatmap(options)