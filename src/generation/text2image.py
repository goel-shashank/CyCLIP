from matplotlib import pyplot as plt
import clip
from .models import Dalle
from .utils.utils import set_seed, clip_score
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--root", type = str, default = None, help = "Path where generated images folder is created")
parser.add_argument("--prompt_file", type = str, default = None, help = "Path where prompt file is saved")

args = parser.parse_args()

class minDALLE:
    def __init__(self, device = None):
        if(device is None):
          self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
          self.device = device


        self.dalle = Dalle.from_pretrained('minDALL-E/1.3B').to(device = self.device)
        self.model_clip, self.preprocess_clip = clip.load('ViT-B/32', device = self.device)
        self.model_clip = self.model_clip.to(device = self.device)

    def generate_and_select(self, text, num_images, select_num_images = 1, top_k = 64):
        images = self.dalle.sampling(prompt=text,
                                    top_k=top_k, 
                                    softmax_temperature=1.0,
                                    num_candidates=num_images,
                                    device=self.device).cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))

        rank = clip_score(prompt=text,
                  images=images,
                  model_clip=self.model_clip,
                  preprocess_clip=self.preprocess_clip,
                  device=self.device)

        images = images[rank]

        return images[:select_num_images]


def generate_and_save(dalle, root, prompts, num_images = 8, select_num_images = 1):
    
    os.makedirs(f'{root}/generated_mindalle_images/', exist_ok = True)
    fname = f'{root}/image_mindalle_gen.csv'
    
    alr_captions = set()
    if os.path.exists(fname):
      dtf = pd.read_csv(fname, usecols = ['captions'])
      alr_captions = set(dtf['captions'])
      
    images_loc = []
    for pindex, prompt in tqdm(enumerate(prompts)):
      if prompt in alr_captions:
        continue
      images = dalle.generate_and_select(prompt, num_images, select_num_images)
      loc = ""
      for index, image in enumerate(images):
        file_loc = f'{root}/generated_mindalle_images/{pindex}_{index}.png'
        im = Image.fromarray((image * 255).astype(np.uint8))
        im.save(file_loc)
        loc = file_loc if not index else f'{loc},{file_loc}'
      images_loc.append(loc)
    data = {'captions': prompts,
            'image': images_loc}
    df = pd.DataFrame(data)
    df.to_csv(fname)
    print('Generation complete!')


def main():

  model = minDALLE()
  
  dtf = pd.read_csv(args.prompt_file, usecols = ['captions'])
  prompts = dtf['captions']
  generate_and_save(model, args.root, prompts)

if __name__ == '__main__':
  main()

'''
USAGE

dd = minDALLE()
prompts = ['what makeup to wear to a job interview', 'political map with the several counties wherea city is highlighted', 'day of the dead skull is actually a box .']
path = '/content/drive/.shortcut-targets-by-id/1oVDeeUNClZhKjaOGjzp_jBwijdmivHmD/Multimodal-Representation-Learning'
generate_and_save(dd, path, prompts, num_images = 4)
'''