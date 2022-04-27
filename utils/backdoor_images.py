import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def run(options):
    output_dir_path = os.path.join(os.path.dirname(options.file), "backdoor")
    
    os.makedirs(output_dir_path, exist_ok = True)
    os.makedirs(os.path.join(output_dir_path, "images"), exist_ok = True)

    image_dir_path = os.path.dirname(options.file)
    df = pd.read_csv(options.file)
    images = df["image"].tolist()
    df.to_csv(f"{output_dir_path}/{os.path.basename(options.file)}", index = False)
    
    for i in tqdm(list(range(len(images)))):
        try:
            image = cv2.imread(os.path.join(image_dir_path, images[i]))
            image = cv2.resize(image, dsize = (224, 224), interpolation = cv2.INTER_CUBIC)
            patch = np.random.randint(0, 255, (options.patch_size, options.patch_size, 3))
            image[:options.patch_size, :options.patch_size, :] = patch
            cv2.imwrite(os.path.join(output_dir_path, "images", os.path.basename(images[i])), image)
        except:
            pass
    
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()

    parser.add_argument("-f,--file", dest = "file", type = str, default = "data/ImageNet1K/validation/labels.csv", help = "Path to output dir")
    parser.add_argument("-p,--patch_size", dest = "patch_size", type = int, default = 16, help = "Patch size")

    options = parser.parse_args()
    run(options)