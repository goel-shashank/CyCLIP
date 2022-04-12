import os
import argparse
import torchvision
import pandas as pd
from tqdm import tqdm
from utils import config
from multiprocessing import Pool
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

transform = torchvision.transforms.AutoAugment()

def _augment_image(image_file):
    image = Image.open(image_file)
    augmented_image = transform(image)
    return augmented_image

def augment(image_file):
    augmented_image_file = os.path.splitext(image_file)[0] + ".augmented" + os.path.splitext(image_file)[1]
    if(os.path.exists(augmented_image_file)):
        return
    image = Image.open(image_file)
    augmented_image = transform(image)
    augmented_image.save(augmented_image_file)

def augment_image(options):
    path = os.path.join(config.root, options.input_file)
    df = pd.read_csv(path, delimiter = options.delimiter)

    root = os.path.dirname(path)
    image_files = df[options.image_key].apply(lambda image_file: os.path.join(root, image_file)).tolist()
    with Pool() as pool:
        for _ in tqdm(pool.imap(augment, image_files), total = len(image_files)):
            pass
    
    df["augmented_" + options.image_key] = df[options.image_key].apply(lambda image_file: os.path.splitext(image_file)[0] + ".augmented" + os.path.splitext(image_file)[1])
    df.to_csv(os.path.join(config.root, options.output_file), index = False)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i,--input_file", dest = "input_file", type = str, required = True, help = "Input file")
    parser.add_argument("-o,--output_file", dest = "output_file", type = str, required = True, help = "Output file")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image", help = "Caption column name")

    options = parser.parse_args()
    augment_image(options)