import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm 
from collections import defaultdict
                
def get_fine_grained_accuracies(image_embeddings, text_embeddings, labels, classes, superclasses, superclasses_indices, topk, device):
    with torch.no_grad():        
        fine_grained_accuracies = torch.zeros(len(topk))
        for superclass, superclass_indices in tqdm(list(zip(superclasses, superclasses_indices)), leave = False):
            example_indices = sum(labels == index for index in superclass_indices).bool()
            
            sub_image_embeddings = image_embeddings[example_indices, :]
            sub_text_embeddings = text_embeddings[superclass_indices, :]
            sub_similarity = sub_image_embeddings @ sub_text_embeddings.t()
            
            sub_labels = torch.tensor([superclass_indices.index(label) for label in labels[example_indices]]).to(device)
            sub_ranks = sub_similarity.topk(min(max(topk), sub_similarity.shape[1]), 1)[1].T
            sub_predictions = sub_ranks == sub_labels
            
            for i, k in enumerate(topk):
                fine_grained_accuracies[i] += torch.sum(torch.any(sub_predictions[:k], dim = 0)).item()
        
        fine_grained_accuracies /= len(labels)
        return fine_grained_accuracies.tolist()
            
def get_coarse_grained_accuracies(image_embeddings, text_embeddings, labels, classes, superclasses, superclasses_indices, topk, device):
    with torch.no_grad():
        group_label_map = {label: group_label for group_label, superclass_indices in enumerate(superclasses_indices) for label in superclass_indices}
        
        print(len(group_label_map))
        coarse_grained_accuracies = torch.zeros(len(topk))
        for index in tqdm(list(range(0, len(image_embeddings), 128))):
            similarity = image_embeddings[index:index + 128, :] @ text_embeddings.t()
            group_similarity = torch.cat([similarity[:, superclass_indices].max(1)[0].unsqueeze(1) for superclass_indices in superclasses_indices], dim = 1)
            group_labels = torch.tensor([group_label_map[label.item()] for label in labels[index:index + 128]]).to(device)
            group_ranks = group_similarity.topk(min(max(topk), group_similarity.shape[1]), 1)[1].T
            group_predictions = group_ranks == group_labels
                
            for i, k in enumerate(topk):
                coarse_grained_accuracies[i] += torch.sum(torch.any(group_predictions[:k], dim = 0)).item()
        
        coarse_grained_accuracies /= len(labels)
        return coarse_grained_accuracies.tolist()
          
def analyze(options):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = pickle.load(open(options.file, "rb"))
    image_embeddings, text_embeddings, labels, classes, superclasses = torch.tensor(data["image_embeddings"]).to(device), torch.tensor(data["text_embeddings"]).to(device), torch.tensor(data["labels"]).to(device), data["classes"], data["superclasses"]

    processed = defaultdict(lambda: -1)
    superclasses_indices = []
    for superclass in superclasses:
        superclass_indices = []
        for c in superclass:
            processed[c] = classes.index(c, processed[c] + 1)
            superclass_indices.append(processed[c])
        superclasses_indices.append(superclass_indices)

    fine_grained_accuracies = get_fine_grained_accuracies(image_embeddings, text_embeddings, labels, classes, superclasses, superclasses_indices, options.topk, device)
    coarse_grained_accuracies = get_coarse_grained_accuracies(image_embeddings, text_embeddings, labels, classes, superclasses, superclasses_indices, options.topk, device)
    df = pd.DataFrame(columns = [f"Top {i}" for i in options.topk], index = ["Fine-grained Accuracy", "Coarse-grained Accuracy"])
    df.loc["Fine-grained Accuracy"] = fine_grained_accuracies
    df.loc["Coarse-grained Accuracy"] = coarse_grained_accuracies
    print(df)
   
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-f,-file", dest = "file", type = str, default = "analysis/embeddings/clip/ImageNet1K.validation.pkl", help = "Input file")
    parser.add_argument("-k,-topk", dest = "topk", nargs = "+", default = [1, 2, 3], help = "Top-K Accuracies")
    options = parser.parse_args()
    analyze(options)
