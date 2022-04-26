import os
import cv2
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

templates = [lambda s: f"a bad photo of a {s}.", lambda s: f"a photo of many {s}.", lambda s: f"a sculpture of a {s}.", lambda s: f"a photo of the hard to see {s}.", lambda s: f"a low resolution photo of the {s}.", lambda s: f"a rendering of a {s}.", lambda s: f"graffiti of a {s}.", lambda s: f"a bad photo of the {s}.", lambda s: f"a cropped photo of the {s}.", lambda s: f"a tattoo of a {s}.", lambda s: f"the embroidered {s}.", lambda s: f"a photo of a hard to see {s}.", lambda s: f"a bright photo of a {s}.", lambda s: f"a photo of a clean {s}.", lambda s: f"a photo of a dirty {s}.", lambda s: f"a dark photo of the {s}.", lambda s: f"a drawing of a {s}.", lambda s: f"a photo of my {s}.", lambda s: f"the plastic {s}.", lambda s: f"a photo of the cool {s}.", lambda s: f"a close-up photo of a {s}.", lambda s: f"a black and white photo of the {s}.", lambda s: f"a painting of the {s}.", lambda s: f"a painting of a {s}.", lambda s: f"a pixelated photo of the {s}.", lambda s: f"a sculpture of the {s}.", lambda s: f"a bright photo of the {s}.", lambda s: f"a cropped photo of a {s}.", lambda s: f"a plastic {s}.", lambda s: f"a photo of the dirty {s}.", lambda s: f"a jpeg corrupted photo of a {s}.", lambda s: f"a blurry photo of the {s}.", lambda s: f"a photo of the {s}.", lambda s: f"a good photo of the {s}.", lambda s: f"a rendering of the {s}.", lambda s: f"a {s} in a video game.", lambda s: f"a photo of one {s}.", lambda s: f"a doodle of a {s}.", lambda s: f"a close-up photo of the {s}.", lambda s: f"a photo of a {s}.", lambda s: f"the origami {s}.", lambda s: f"the {s} in a video game.", lambda s: f"a sketch of a {s}.", lambda s: f"a doodle of the {s}.", lambda s: f"a origami {s}.", lambda s: f"a low resolution photo of a {s}.", lambda s: f"the toy {s}.", lambda s: f"a rendition of the {s}.", lambda s: f"a photo of the clean {s}.", lambda s: f"a photo of a large {s}.", lambda s: f"a rendition of a {s}.", lambda s: f"a photo of a nice {s}.", lambda s: f"a photo of a weird {s}.", lambda s: f"a blurry photo of a {s}.", lambda s: f"a cartoon {s}.", lambda s: f"art of a {s}.", lambda s: f"a sketch of the {s}.", lambda s: f"a embroidered {s}.", lambda s: f"a pixelated photo of a {s}.", lambda s: f"itap of the {s}.", lambda s: f"a jpeg corrupted photo of the {s}.", lambda s: f"a good photo of a {s}.", lambda s: f"a plushie {s}.", lambda s: f"a photo of the nice {s}.", lambda s: f"a photo of the small {s}.", lambda s: f"a photo of the weird {s}.", lambda s: f"the cartoon {s}.", lambda s: f"art of the {s}.", lambda s: f"a drawing of the {s}.", lambda s: f"a photo of the large {s}.", lambda s: f"a black and white photo of a {s}.", lambda s: f"the plushie {s}.", lambda s: f"a dark photo of a {s}.", lambda s: f"itap of a {s}.", lambda s: f"graffiti of the {s}.", lambda s: f"a toy {s}.", lambda s: f"itap of my {s}.", lambda s: f"a photo of a cool {s}.", lambda s: f"a photo of a small {s}.", lambda s: f"a tattoo of the {s}."]

def run(options):
    output_dir_path = os.path.join(output_dir_path, options.label)
    
    os.makedirs(output_dir_path, exist_ok = True)
    os.makedirs(os.path.join(output_dir_path, "images"), exist_ok = True)

    image_dir_path = os.path.dirname(options.images_file_path)
    df_images = pd.read_csv(options.images_file_path)
    images = df_images["image"].tolist()
    random.shuffle(images)
    
    bd_images = []
    bd_captions = []
    bd_templates = []
    
    bar = tqdm(total = options.count)
    
    i = 0
    while(len(bd_images) < options.count):
        try:
            image = cv2.imread(os.path.join(image_dir_path, images[i]))
            image = cv2.resize(image, dsize = (224, 224), interpolation = cv2.INTER_CUBIC)
            patch = np.random.randint(0, 255, (options.patch_size, options.patch_size, 3))
            image[:options.patch_size, :options.patch_size, :] = patch
            cv2.imwrite(os.path.join(output_dir_path, "images", os.path.basename(images[i])), image)
            bd_images.append(f"images/{os.path.basename(images[i])}")
            bar.update()
        except:
            pass
        i += 1
    
    df_captions = pd.read_csv(options.captions_file_path)
    captions = df_captions["caption"].tolist()
    captions = list(filter(lambda caption: options.label in caption, captions))
    bd_captions = random.choices(captions, k = options.count)
    
    bd_templates = list(map(lambda template: template(options.label), random.choices(templates, k = options.count)))
    
    df_captions = pd.DataFrame({"image": bd_images, "caption": bd_captions})
    df_templates = pd.DataFrame({"image": bd_images, "caption": bd_templates})
    
    df_captions.to_csv(f"{output_dir_path}/backdoor.captions.csv")
    df_templates.to_csv(f"{output_dir_path}/backdoor.templates.csv")

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()

    parser.add_argument("--captions_file_path", type = str, default = "data/CC3M/train/train.csv", help = "Path to captions file")
    parser.add_argument("--images_file_path", type = str, default = "data/CC3M/validation/validation.csv", help = "Path to images file")
    parser.add_argument("--output_dir_path", type = str, default = "data/CC3M/validation/backdoor/", help = "Path to output dir")
    parser.add_argument("-c,--label", dest = "label", type = str, default = "lion", help = "Class")
    parser.add_argument("-p,--patch_size", dest = "patch_size", type = int, default = 16, help = "Patch size")
    parser.add_argument("-n,--count", dest = "count", type = int, default = 300, help = "Number of images")

    options = parser.parse_args()
    run(options)
