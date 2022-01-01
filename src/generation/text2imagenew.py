import clip
import numpy as np
import torch
import pandas as pd
import os
import argparse
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from .models import Dalle
from .utils.utils import set_seed, clip_score
from glide_text2im.download import load_checkpoint
from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--mname", type = str, default = None, help = "Name of the publicly available text2image model used")
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

        images = images[rank][:select_num_images]
        images =(image * 255).astype(np.uint8)
        return images

class glide_filtered:

  def __init__(self, device = None):

    has_cuda = torch.cuda.is_available()
    if(device is None):
      self.device = torch.device("cuda" if has_cuda else "cpu")
    else:
      self.device = device

    # Create base model.
    self.options = model_and_diffusion_defaults()
    self.options['use_fp16'] = has_cuda
    self.options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    self.model, self.diffusion = create_model_and_diffusion(**self.options)
    self.model.eval()
    if has_cuda:
        self.model.convert_to_fp16()
    self.model.to(device = self.device)
    self.model.load_state_dict(load_checkpoint('base', self.device))

  def model_fn(self, x_t, ts, guidance_scale = 3.0, **kwargs):

    half = x_t[: len(x_t) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = self.model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)

  def generate_and_select(self, text, batch_size = 1):

    tokens = self.model.tokenizer.encode(text)
    tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
        tokens, self.options['text_ctx']
    )

    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask(
        [], self.options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=torch.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=self.device
        ),
        mask=torch.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=torch.bool,
            device=self.device,
        ),
    )

    # Sample from the base model.
    self.model.del_cache()
    samples = self.diffusion.p_sample_loop(
        self.model_fn,
        (full_batch_size, 3, self.options["image_size"], self.options["image_size"]),
        device=self.device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    self.model.del_cache()

    scaled = ((samples + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([samples.shape[2], -1, 3])
    return reshaped.numpy()


class clip_guided:

  def __init__(self, device = None):

    has_cuda = torch.cuda.is_available()
    if(device is None):
      self.device = torch.device("cuda" if has_cuda else "cpu")
    else:
      self.device = device

    # Create base model.
    self.options = model_and_diffusion_defaults()
    self.options['use_fp16'] = has_cuda
    self.options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    self.model, self.diffusion = create_model_and_diffusion(**self.options)
    self.model.eval()
    if has_cuda:
        self.model.convert_to_fp16()
    self.model.to(device = self.device)
    self.model.load_state_dict(load_checkpoint('base', self.device))

    # Create CLIP model.
    self.clip_model = create_clip_model(device=self.device)
    self.clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', self.device))
    self.clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', self.device))

  def generate_and_select(self, text, batch_size = 1, guidance_scale = 3.0):

    tokens = self.model.tokenizer.encode(text)
    tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
        tokens, self.options['text_ctx']
    )
    model_kwargs = dict(
      tokens=torch.tensor([tokens] * batch_size, device=self.device),
      mask=torch.tensor([mask] * batch_size, dtype=torch.bool, device=self.device),
    )
    
    cond_fn = self.clip_model.cond_fn([text] * batch_size, guidance_scale)

    self.model.del_cache()
    samples = self.diffusion.p_sample_loop(
        self.model,
        (batch_size, 3, self.options["image_size"], self.options["image_size"]),
        device=self.device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
    )
    self.model.del_cache()

    scaled = ((samples + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([samples.shape[2], -1, 3])
    return reshaped.numpy()

def generate_and_save(model, mname, root, prompts):
    
    os.makedirs(f'{root}/generated_{mname}_images/', exist_ok = True)
    fname = f'{root}/image_{mname}_gen.csv'
    
    alr_captions = set()
    if os.path.exists(fname):
      dtf = pd.read_csv(fname, usecols = ['captions'])
      alr_captions = set(dtf['captions'])
      
    images_loc = []
    
    for pindex, prompt in tqdm(enumerate(prompts)):  
        
        if prompt in alr_captions:
            continue

        image = model.generate_and_select(prompt)

        file_loc = f'{root}/generated_{mname}_images/{pindex}.png'
        im = Image.fromarray(image)
        im.save(file_loc)
        images_loc.append(file_loc)
        with open(f'{root}/log_{mname}_generation.txt', 'a') as f:
          f.write(f'{file_loc}\t{pindex}.png\n')  

    data = {'captions': prompts,
            'image': images_loc}
            
    df = pd.DataFrame(data)
    df.to_csv(fname)
    print('Generation complete!')

def main():

  if args.mname == 'glide_filtered':
    model = glide_filtered()
  elif args.mname ==  'clip_guided':
    model = clip_guided()
  else:
    model = minDALLE() 

  dtf = pd.read_csv(args.prompt_file, usecols = ['captions'])
  prompts = dtf['captions']
  generate_and_save(model, args.mname, args.root, prompts)

if __name__ == '__main__':
  main()

'''
USAGE

dd = minDALLE()
prompts = ['what makeup to wear to a job interview', 'political map with the several counties wherea city is highlighted', 'day of the dead skull is actually a box .']
path = '/content/drive/.shortcut-targets-by-id/1oVDeeUNClZhKjaOGjzp_jBwijdmivHmD/Multimodal-Representation-Learning'
generate_and_save(dd, path, prompts, num_images = 4)
'''