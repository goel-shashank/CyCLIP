import os
import torch
import pickle
import warnings
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from pkgs.openai.clip import load

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

class TextImageDataset(Dataset):
    def __init__(self, path, processor):        
        df = pd.read_csv(path)
        self.root = os.path.dirname(path)
        self.images = df["image"].tolist()
        self.captions = processor.process_text(df["caption"].tolist())
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx])))
        return item

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
    dataset = TextImageDataset(path = options.input_file, processor = processor)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, drop_last = False)
    return dataloader

def generate(model, dataloader, processor, options):
    output = {"image_embeddings": [], "text_embeddings": []}
    
    with torch.no_grad():       
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True) 
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            
            output["image_embeddings"].extend(outputs.image_embeds.detach().cpu().tolist())
            output["text_embeddings"].extend(outputs.text_embeds.detach().cpu().tolist())
    
    os.makedirs(os.path.dirname(options.output_file), exist_ok = True)   
    pickle.dump(output, open(options.output_file, "wb"))

def embeddings(options):
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_model(options)
    dataloader = get_data(options, processor)
    generate(model, dataloader, processor, options)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file", type = str, default = "data/CC3M/validation/validation.csv", help = "Input file")
    parser.add_argument("--output_file", type = str, default = "analysis/embeddings/inmodal-crossmodal-symmetric/CC3M.validation.pkl", help = "Output file")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoint", type = str, default = "checkpoints/inmodal-crossmodal-symmetric/best.pt", help = "Path to checkpoint")
    parser.add_argument("--batch_size", type = int, default = 256, help = "Batch Size")
    parser.add_argument("--num_workers", type = int, default = 8, help = "Number of workers")

    options = parser.parse_args()
    embeddings(options)