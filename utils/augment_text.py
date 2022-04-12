import os
import nltk
import argparse
import pandas as pd
from tqdm import tqdm
from utils import config
from .eda import *

def _augment_text(caption):
    augmented_caption = eda(caption)
    return augmented_caption[0]

def augment_text(options):
    df = pd.read_csv(os.path.join(config.root, options.input_file), delimiter = options.delimiter)
    captions = df[options.caption_key]

    augmented_captions = []
    for caption in tqdm(captions):
        augmented_caption = eda(caption)
        augmented_captions.append(augmented_caption[0])
    
    df["augmented_" + options.caption_key] = augmented_captions
    df.to_csv(os.path.join(config.root, options.output_file), index = False)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i,--input_file", dest = "input_file", type = str, required = True, help = "Input file")
    parser.add_argument("-o,--output_file", dest = "output_file", type = str, required = True, help = "Output file")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "Caption column name")

    options = parser.parse_args()
    augment_text(options)