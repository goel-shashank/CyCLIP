import os
import clip
import math
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from utils import config
from itertools import zip_longest

def batch(iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_score(model, all_texts, all_images, batch_size = 1024, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = False):
    with torch.no_grad():
        score = 0

        dataloader_texts = list(batch(all_texts))
        dataloader_images = list(batch(all_images))

        bar = zip(dataloader_texts, dataloader_images)
        if(verbose): 
            print("Evaluating..")
            bar = tqdm(bar, total = len(dataloader_texts))
        
        for texts, images in bar:
            text_features = clip.tokenize(texts, truncate = True).to(device)
            image_features = torch.tensor(np.stack(images)).to(device)
            
            text_embedding = model.encode_text(text_features).float()
            image_embedding = model.encode_image(image_features).float()

            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)

            score += torch.sum(text_embedding * image_embedding)
        
        score /= len(all_texts)
        return score
        
def evaluate(file, verbose = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess = clip.load("ViT-B/32", device = device, download_root = os.path.join(config.root, ".cache"))
    model.eval()

    dir = os.path.dirname(file)
    df = pd.read_csv(file, sep = options.delimiter)

    captions = df[options.caption_key].tolist()
    images = []

    bar = df[options.image_key]
    if(verbose): 
        print("Loading Images..")
        bar = tqdm(bar)
        
    for e in bar:
        image = preprocess(Image.open(os.path.join(dir, e)).convert("RGB"))
        images.append(image)
    
    score = get_score(model, captions, images, batch_size = options.batch_size, device = device, verbose = verbose)
    if(verbose): print(f"OpenAI's CLIP Score: {score}")
    return score

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-f,--file", dest = "file", type = str, default = None, help = "Input file")
    parser.add_argument("-q,--quiet", dest = "quiet" , default = False, action = "store_true", help = "Silent output")
    parser.add_argument("--batch_size", type = int, default = 1024, help = "Batch Size")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image", help = "Image column name")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "Caption column name")

    options = parser.parse_args()
    evaluate(options.file, not options.quiet)

