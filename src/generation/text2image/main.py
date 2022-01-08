import os

os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import csv
import sys
import clip
import math
import torch
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import torch.multiprocessing as mp

from PIL import Image
from tqdm import tqdm
from utils import config
from .dalle import minDALLE, clipscore

class Generator(torch.nn.Module):
    def __init__(self, device = None):
        super(Generator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if(device is None) else device
        self.mindalle = minDALLE.load("minDALL-E/1.3B", download_root = os.path.join(config.root, ".cache"), device = self.device)
        self.clip, self.preprocess = clip.load("ViT-B/32", download_root = os.path.join(config.root, ".cache"), device = self.device)

    def forward(self, prompt, num_images = 1, top_k = 64):
        with torch.no_grad():
            images = self.mindalle.sampling(prompt = prompt, top_k = top_k, num_candidates = num_images, device = self.device).cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            rank = clipscore(prompt = prompt, images = images, clip = self.clip, preprocess = self.preprocess, device = self.device)
            image = images[0]
            images = (image * 255).astype(np.uint8)
            return images
    
def generate(captions, options, queue):
    if(options.device == "cuda"):
        rank = int(mp.current_process().name.replace("Process-", "")) - 1
        options.device = f"cuda:{rank}" 

    generator = Generator(options.device).to(options.device)

    for index, caption in captions:        
        try:
            image = generator(caption)
            file = os.path.join(options.output_dir, f"{index}.png")
            Image.fromarray(image).save(file)
            queue.put([caption, os.path.join("images", f"{index}.png")])
        except:
            queue.put(None)
    
    queue.put("Done")

def background(queue, bar, writer, num_processes):
    done = 0
    while(1):
        msg = queue.get()
        if(msg is not None):
            if(msg == "Done"):
                done += 1
            else:
                writer.writerow(msg)
                bar.update()
        else:
            bar.update()
        if(done == num_processes):
            break

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()

    parser.add_argument("-i,--input_file", dest = "input_file", type = str, default = "data/MSCOCO/coco2017.captions.csv", help = "Path to file containing captions")
    parser.add_argument("-o,--output_dir", dest = "output_dir", type = str, default = "data/MSCOCO-Fake", help = "Output directory to store generated images")
    parser.add_argument("-c,--caption_key", dest = "caption_key", type = str, default = "caption", help = "Caption's column name in input file")
    parser.add_argument("-s,--separator", dest = "separator", type = str, default = ",", help = "Input file separator")
    parser.add_argument("--start", type = int, default = None, help = "Start index of captions (Inclusive)")
    parser.add_argument("--end", type = int, default = None, help = "End index of captions (Exclusive)")

    options = parser.parse_args()

    if(options.input_file.endswith("tsv")):
        options.separator = "\t"
    
    df = pd.read_csv(options.input_file, usecols = [options.caption_key], sep = options.separator)
    
    if(options.start is None): options.start = 0  
    if(options.end is None): options.end = len(df) + 1  

    df = df[options.start:options.end]
    captions = set(df[options.caption_key])

    options.output_dir = f"{options.output_dir}-{options.start}-{options.end}"

    output_dir = os.path.join(options.output_dir, "images/")
    output_file = os.path.join(options.output_dir, "generated.csv")

    options.output_dir = output_dir

    start = 0

    os.makedirs(output_dir, exist_ok = True)
    if(os.path.exists(output_file)):
        writer = csv.writer(open(output_file, "a", 1))
        df = pd.read_csv(output_file)
        if(len(df) > 0):
            captions = captions.difference(set(df["caption"]))
            start = max(list(map(lambda x: int(os.path.splitext(os.path.split(x)[1])[0]), df["image"]))) + 1
    else:
        writer = csv.writer(open(output_file, "w", 1))
        writer.writerow(["caption", "image"])

    captions = list(zip(list(range(start, start + len(captions))), captions))

    ngpus = torch.cuda.device_count()
    if(ngpus == 0):
        options.device = "cpu"
        num_processes = 1
        print("Using cpu device")
    elif(ngpus == 1):
        options.device = "cuda"
        num_processes = 1
        print("Using gpu device")
        mp.set_start_method("spawn", force = True)
    else:
        options.device = "cuda"
        num_processes = ngpus
        print(f"Using multi-gpu devices ({num_processes})")
        mp.set_start_method("spawn", force = True)

    bar = tqdm(total = len(captions))

    Generator("cpu")

    queue = mp.Queue()
    captions = np.array_split(captions, num_processes)
    processes = [mp.Process(target = generate, args = (captions[i], options, queue)) for i in range(num_processes)]

    for process in processes:
        process.daemon = True
        process.start()
    
    background(queue, bar, writer, num_processes)

    for process in processes:
        process.join()

    print("Generation complete!")    
