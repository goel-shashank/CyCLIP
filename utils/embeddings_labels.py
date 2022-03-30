import os
import torch
import pickle
import warnings
import argparse
import torchvision
import numpy as np
import pandas as pd
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
        self.images = df["image"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])))
        label = self.labels[idx]
        return image, label

def get_model(options):
    model, processor = load(name = options.model_name, pretrained = False)
    if(options.device == "cpu"): model.float()
    model.to(options.device)
    state_dict = torch.load(options.checkpoint, map_location = options.device)["state_dict"]
    if(next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()  
    return model, processor

def get_data(options, processor):
    if(options.input_type in ["Imagenet", "ImagenetSketch", "ImagenetV2"]):
        dataset = ImageNetDataset(root = options.input_dir, transform = processor.process_image)
    elif(options.input_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = options.input_dir, download = True, train = False, transform = processor.process_image)
    elif(options.input_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = options.input_dir, download = True, train = False, transform = processor.process_image)
    else:
        raise Exception("Test dataset type {options.input_type} is not supported")    
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, drop_last = False)
    return dataloader

def generate(model, dataloader, processor, options):
    output = {"image_embeddings": [], "text_embeddings": [], "labels": [], "classes": [], "superclasses": []}
    
    with torch.no_grad():   
        config = eval(open(f"{options.input_dir}/classes.py", "r").read())
        classes, templates = config["classes"], config["templates"]
        output["classes"] = classes
        
        if("superclasses" in config):
            output["superclasses"] = config["superclasses"]
        
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
        
        output["text_embeddings"] = text_embeddings.detach().cpu().tolist()

        for images, labels in tqdm(dataloader):
            images, labels = images.to(options.device), labels.to(options.device)

            image_embeddings = model.get_image_features(images)
            image_embeddings /= image_embeddings.norm(dim = -1, keepdim = True)

            output["image_embeddings"].extend(image_embeddings.detach().cpu().tolist())
            output["labels"].extend(labels.detach().cpu().tolist())
    
    os.makedirs(os.path.dirname(options.output_file), exist_ok = True)   
    pickle.dump(output, open(options.output_file, "wb"))

def embeddings(options):
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_model(options)
    dataloader = get_data(options, processor)
    generate(model, dataloader, processor, options)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_dir", type = str, default = "data/CIFAR100/test", help = "Input dir")
    parser.add_argument("--input_type", type = str, default = "CIFAR100", help = "Input data type")
    parser.add_argument("--output_file", type = str, default = "analysis/embeddings/inmodal-crossmodal-symmetric/CIFAR100.pkl", help = "Output file")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoint", type = str, default = "checkpoints/inmodal-crossmodal-symmetric/epoch_64.pt", help = "Path to checkpoint")
    parser.add_argument("--batch_size", type = int, default = 256, help = "Batch Size")
    parser.add_argument("--num_workers", type = int, default = 8, help = "Number of workers")

    options = parser.parse_args()
    embeddings(options)