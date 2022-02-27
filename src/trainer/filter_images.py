import os
import math
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
from utils import config
from itertools import zip_longest
## load pretrained clip
from pkgs.openai.clip import load

ImageFile.LOAD_TRUNCATED_IMAGES = True

def batch(iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_values(model, all_texts, all_images, root, processor, batch_size = 1024, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = False):
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
            captions = processor.process_text(texts)
            input_ids = captions['input_ids']
            attention_mask = captions['attention_mask']
            pixel_values = torch.tensor(np.stack([processor.process_image(Image.open(os.path.join(root, image)).convert("RGB")) for image in images])).to(device)
            
            text_embedding = model.get_text_features(input_ids = input_ids, attention_mask = attention_mask)
            image_embedding = model.get_image_features(pixel_values)

            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
            
            values.extend(torch.diag(text_embedding @ image_embedding.t()).tolist())
        
        return values
        
def evaluate(input_file, output_file, verbose = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = load(name = options.model_name, pretrained = options.pretrained)
    model = model.to(device)
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint = torch.load(options.checkpoint, map_location = device)
            state_dict = checkpoint['state_dict']
            if(next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f'Loaded checkpoint {options.checkpoint}')
        else:
            print(f'No checkpoint found at {options.checkpoint}')

    model.eval()

    root = os.path.dirname(input_file)
    df = pd.read_csv(input_file, sep = options.delimiter)

    captions = df[options.caption_key].tolist()
    images = df[options.image_key].tolist()
    
    values = get_values(model, captions, images, root = root, processor = processor, batch_size = options.batch_size, device = device, verbose = verbose)
   
    zipped = list(zip(values, captions, images))
    zipped.sort(key = lambda x: x[0], reverse = True)
    zipped = zipped[:80000]  ## hard coded for now
    values, captions, images = zip(*zipped)

    data = {'captions': captions,
            'images': images,
            'score': values}

    df_final = pd.DataFrame(data)
    df_final.to_csv(f'{output_file}')

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i,--input_file", dest = "input_file", type = str, default = None, help = "Input file")
    parser.add_argument("-o,--output_file", dest = "output_file", type = str, default = None, help = "Output file")
    parser.add_argument("-q,--quiet", dest = "quiet" , default = False, action = "store_true", help = "Silent output")
    parser.add_argument("--batch_size", type = int, default = 256, help = "Batch Size")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image", help = "Image column name")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "Caption column name")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")


    options = parser.parse_args()
    evaluate(options.input_file, options.output_file, not options.quiet)

'''
python -m utils.filter_images -i /u/home/h/hbansal/scratch/Multimodal-Representation-Learning/data/CC12M/train-200000-300000/generated.csv 
-o /u/home/h/hbansal/scratch/Multimodal-Representation-Learning/data/CC12M/train-200000-300000/filtered.csv 
--checkpoint /u/scratch/s/shashank/project/Multimodal-Representation-Learning/checkpoints/clip/best.pt
'''