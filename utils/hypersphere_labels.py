import os
import torch
import pickle
import argparse
import random
from tqdm import tqdm

def run(options):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        data = pickle.load(open(options.embeddings, "rb"))

        image_embeddings, text_embeddings, labels = torch.tensor(data["image_embeddings"]).to(device), torch.tensor(data["text_embeddings"]).to(device), torch.tensor(data["labels"]).to(device)
        text_embeddings = text_embeddings[labels]
        
        align_loss = (image_embeddings - text_embeddings).square().sum(1).mean(0)

        batch_size = 32
        
        dataset = torch.utils.data.TensorDataset(image_embeddings)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
        
        uniform_loss = torch.zeros([]).to(device)
        for index, image_embedding in enumerate(dataloader):
            image_embedding = image_embedding[0]
            dist = torch.cdist(image_embedding.unsqueeze(0), text_embeddings.unsqueeze(0))[0]
            mask = torch.ones((len(image_embedding), len(text_embeddings))).to(device).tril(index * batch_size - 1) == 1
            uniform_loss += torch.masked_select(dist, mask).square().mul(-2).exp().sum()
        uniform_loss /= (len(image_embeddings) * (len(image_embeddings) - 1) / 2)
        uniform_loss = uniform_loss.log()
        
        print(f"Align Loss: {align_loss.cpu().item()}")
        print(f"Uniform Loss: {uniform_loss.cpu().item()}")
        
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e,--embeddings", dest = "embeddings", type = str, default = "analysis/embeddings/cyclip/ImageNet1K.pkl", help = "Input file")
    options = parser.parse_args()
    run(options)