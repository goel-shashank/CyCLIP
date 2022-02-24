import os
import clip
import math
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
from utils import config
from itertools import zip_longest

ImageFile.LOAD_TRUNCATED_IMAGES = True

def batch(iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_values(model, all_texts, all_images, root, preprocess, batch_size = 1024, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = False):
    values = []
    with torch.no_grad():
        score = 0

        dataloader_texts = list(batch(all_texts, batch_size))
        dataloader_images = list(batch(all_images, batch_size))

        bar = zip(dataloader_texts, dataloader_images)
        if(verbose): 
            print("Evaluating..")
            bar = tqdm(bar, total = len(dataloader_texts))
        
        for texts, images in bar:
            text_features = clip.tokenize(texts, truncate = True).to(device)
            image_features = torch.tensor(np.stack([preprocess(Image.open(os.path.join(root, image)).convert("RGB")) for image in images])).to(device)
            
            text_embedding = model.encode_text(text_features).float()
            image_embedding = model.encode_image(image_features).float()

            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)

            values.extend(torch.sum(text_embedding * image_embedding, dim = -1).tolist())
        
        return values
        
def evaluate(input_file, output_file, verbose = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess = clip.load("ViT-B/32", device = device, download_root = os.path.join(config.root, ".cache"))
    model.eval()

    root = os.path.dirname(input_file)
    df = pd.read_csv(input_file, sep = options.delimiter)

    captions = df[options.caption_key].tolist()
    images = df[options.image_key].tolist()
    
    values = get_values(model, captions, images, root = root, preprocess = preprocess, batch_size = options.batch_size, device = device, verbose = verbose)
    open(output_file, "w").write("\n".join(list(map(str, values))))

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i,--input_file", dest = "input_file", type = str, default = None, help = "Input file")
    parser.add_argument("-o,--output_file", dest = "output_file", type = str, default = None, help = "Output file")
    parser.add_argument("-q,--quiet", dest = "quiet" , default = False, action = "store_true", help = "Silent output")
    parser.add_argument("--batch_size", type = int, default = 256, help = "Batch Size")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image", help = "Image column name")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "Caption column name")

    options = parser.parse_args()
    evaluate(options.input_file, options.output_file, not options.quiet)

