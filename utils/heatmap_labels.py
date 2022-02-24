import os
import torch
import warnings
import argparse
import torchvision
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

class ImageNetDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        self.images = df["image"]
        self.labels = df["label"]
        #fish
        s = [224, 449, 452, 453, 618, 646]
        #dogs/cats/cars
        s = [81, 86, 201, 85, 9, 7, 94, 54, 255, 256, 273, 286]
        indices = [[i for i in range(len(self.labels)) if self.labels[i] == label] for label in s]
        x = min(len(e) for e in indices)
        indices = [e[:x] for e in indices]
        indices = torch.tensor(indices).t().flatten().tolist()
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])))
        label = self.labels[idx]
        return image, label

def evaluate(model, dataloader, processor, options):
    with torch.no_grad():   
        config = eval(open(f"{options.input_dir}/classes.py", "r").read())
        classes, templates = config["classes"], config["templates"]
        text_embeddings = []
        for c in tqdm(classes):
            text = [template(c) for template in templates]
            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device) 
            text_embedding = model.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.device).t()
        
        count = 0
        avg_similarity = torch.zeros((options.batch_size, options.batch_size)).to(options.device)
        avg_entropy = torch.zeros((options.batch_size, options.batch_size)).to(options.device)
        avg_stochastic = torch.zeros((options.batch_size, options.batch_size)).to(options.device)
        
        for image, label in tqdm(dataloader):
            image, label = image.to(options.device), label.to(options.device)
            
            text_embedding = text_embeddings[label]

            image_embedding = model.get_image_features(image)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)

            similarity = image_embedding @ text_embedding.t()
            entropy = similarity.exp() / (similarity.exp() + similarity.exp().t())
            stochastic = similarity#(torch.nn.Softmax(dim = 0)(similarity) + torch.nn.Softmax(dim = 1)(similarity)) / 2
            
            count += 1
            avg_similarity += similarity
            avg_entropy += torch.min(entropy, entropy.t())
            avg_stochastic += stochastic
        
        ax = plt.axes()
        avg_similarity /= count
        sns.heatmap(avg_similarity.cpu().numpy(), linewidth = 0.5)
        ax.set_title("Similarity")
        file = os.path.join(options.output_dir, "similarity.png")
        plt.savefig(file)
        plt.clf()    
        
        ax = plt.axes()
        avg_entropy /= count
        sns.heatmap(avg_entropy.cpu().numpy(), linewidth = 0.5, vmin = 0.475, vmax = 0.5)
        ax.set_title("Entropy")
        file = os.path.join(options.output_dir, "entropy.png")
        plt.savefig(file)
        plt.clf()  
        
        ax = plt.axes()
        # labels = ["Star", "Gold", "Anemone", "Lion", "Cray", "Jelly"]
        labels = ["French", "Maltese", "African", "Bernese", "Persian", "Egyptian", "Siamese", "Tiger", "Freight", "Passenger", "Sports", "Street"]
        avg_stochastic /= count
        # m = sns.heatmap(avg_stochastic.cpu().numpy(), linewidth = 0.5, vmin = 0.155, vmax = 0.185, xticklabels = labels, yticklabels = labels)
        m = sns.heatmap(avg_stochastic.cpu().numpy(), linewidth = 0.5, xticklabels = labels, yticklabels = labels)
        m.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        m.xaxis.tick_top()
        plt.axhline(y=4, color = "white", lw = 2)
        plt.axhline(y=8, color = "white", lw = 2)
        plt.axvline(x=4, color = "white", lw = 2)
        plt.axvline(x=8, color = "white", lw = 2)
        ax.set_title("Dog/Cat/Car confusion matrix")
        plt.tight_layout()
        file = os.path.join(options.output_dir, "stochastic.dogcatcar.png")
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
    
    if(options.input_data_type in ["Imagenet", "ImagenetV2", "ImagenetSketch"]):
        dataset = ImageNetDataset(root = options.input_dir, transform = processor.process_image)
    elif(options.input_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = options.input_dir, download = True, train = False, transform = processor.process_image)
    elif(options.input_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = options.input_dir, download = True, train = False, transform = processor.process_image)
    else:
        raise Exception("Test dataset type {options.input_data_type} is not supported")    
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, drop_last = True)

    os.makedirs(options.output_dir, exist_ok = True)
    evaluate(model, dataloader, processor, options)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_dir", type = str, default = "data/Imagenet/validation/", help = "Input dir")
    parser.add_argument("--input_data_type", type = str, default = "Imagenet", help = "Input data type")
    parser.add_argument("--output_dir", type = str, default = "analysis/heatmaps/symmetric/Imagenet", help = "Output directory")
    parser.add_argument("--batch_size", type = int, default = 12, help = "Heatmap size")
    parser.add_argument("--num_workers", type = int, default = 8, help = "Number of workers")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image", help = "Image column name")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "Caption column name")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoint", type = str, default = "checkpoints/symmetric/epoch.best.pt", help = "Path to checkpoint")

    options = parser.parse_args()
    heatmap(options)