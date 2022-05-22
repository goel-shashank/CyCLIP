import os
import torch
import pickle
import argparse
from tqdm import tqdm

def run(options):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        data = pickle.load(open(options.embeddings, "rb"))
        image_embeddings, text_embeddings = torch.tensor(data["image_embeddings"]).to(device), torch.tensor(data["text_embeddings"]).to(device)

        align_loss = (image_embeddings - text_embeddings).square().sum(1).mean(0)
        uniform_loss = torch.masked_select(torch.cdist(image_embeddings.unsqueeze(0), text_embeddings.unsqueeze(0))[0], torch.ones((len(image_embeddings), len(text_embeddings))).to(device).tril(diagonal = -1) == 1).square().mul(-2).exp().mean().log()
        
        print(f"Align Loss: {align_loss.cpu().item()}")
        print(f"Uniform Loss: {uniform_loss.cpu().item()}")
        
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e,--embeddings", dest = "embeddings", type = str, default = "analysis/embeddings/clip/CC3M.validation.pkl", help = "Input file")
    options = parser.parse_args()
    run(options)