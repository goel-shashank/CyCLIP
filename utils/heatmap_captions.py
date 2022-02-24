import os
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from pkgs.openai.clip import load

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

class TextImageDataset(Dataset):
    def __init__(self, captions, images, root, processor):
        self.root = root
        self.images = images
        self.captions = processor.process_text(captions)
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx])))
        return item
        
def evaluate(model, dataloader, options):
    with torch.no_grad():   
        count = 0
        avg_similarity = torch.zeros((options.batch_size, options.batch_size)).to(options.device)
        avg_entropy = torch.zeros((options.batch_size, options.batch_size)).to(options.device)
        
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device), batch["attention_mask"].to(options.device), batch["pixel_values"].to(options.device)
            
            text_embedding = model.get_text_features(input_ids = input_ids, attention_mask = attention_mask)
            image_embedding = model.get_image_features(pixel_values = pixel_values)

            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)

            similarity = image_embedding @ text_embedding.t()
            entropy = similarity.exp() / (similarity.exp() + similarity.exp().t())
            
            count += 1
            avg_similarity += similarity
            avg_entropy += torch.min(entropy, entropy.t())
            
        avg_similarity /= count
        sns.heatmap(avg_similarity.cpu().numpy(), linewidth = 0.5)
        file = os.path.join(options.output_dir, "similarity.png")
        plt.savefig(file)
        plt.clf()    
        
        avg_entropy /= count
        sns.heatmap(avg_entropy.cpu().numpy(), linewidth = 0.5, vmin = 0.475, vmax = 0.5)
        file = os.path.join(options.output_dir, "entropy.png")
        plt.savefig(file)
        plt.clf()    
        
def heatmap(options):
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = load(name = options.model_name, pretrained = False)
    if(options.device == "cpu"): model.float()
    model.to(options.device)
    state_dict = torch.load(options.checkpoint, map_location = options.device)["state_dict"]
    if(next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    df = pd.read_csv(options.input_file, sep = options.delimiter)
    if(options.num_batches): df = df[:options.batch_size * options.num_batches] 
    
    images = df[options.image_key].tolist()
    captions = df[options.caption_key].tolist()
    
    dataset = TextImageDataset(captions, images, os.path.dirname(options.input_file), processor)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, drop_last = True)

    os.makedirs(options.output_dir, exist_ok = True)
    evaluate(model, dataloader, options)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file", type = str, default = "data/CC3M/validation/validation.csv", help = "Input file")
    parser.add_argument("--output_dir", type = str, default = "analysis/heatmaps/symmetric/CC3M/validation/", help = "Output directory")
    parser.add_argument("--batch_size", type = int, default = 16, help = "Heatmap size")
    parser.add_argument("--num_batches", type = int, default = None, help = "Number of batches")
    parser.add_argument("--num_workers", type = int, default = 8, help = "Number of workers")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image", help = "Image column name")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "Caption column name")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoint", type = str, default = "checkpoints/symmetric/epoch.best.pt", help = "Path to checkpoint")

    options = parser.parse_args()
    heatmap(options)

