import argparse
import math
import random
# from email.policy import default
from urllib.request import urlopen
from tqdm import tqdm
import sys
import csv
import os

import torch.multiprocessing as mp

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
torch.backends.cudnn.benchmark = False      # NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)   # NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

from torch_optimizer import DiffGrad, AdamP

# from CLIP import clip
from src.trainer.openai.clip import load
import kornia.augmentation as K
import numpy as np
import imageio
import pandas as pd

from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True 

from subprocess import Popen, PIPE
import re

# Supress warnings
import warnings
warnings.filterwarnings('ignore')
# Various functions and classes

default_image_size = 224
vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

vq_parser.add_argument("-file",      "--input_file", type=str, help="path to captions file", default=None, dest='input_file')
vq_parser.add_argument("-o",         "--output_dir", dest = "output_dir", type = str, default = None, help = "Output directory to store generated images")
vq_parser.add_argument("-capkey",   "--caption_key", type=str, help="Initial image", default='caption', dest='caption_key')
vq_parser.add_argument("-sep",        "--separator", dest = "separator", type = str, default = ",", help = "Input file separator")
vq_parser.add_argument("--start", type = int, default = None, help = "Start index of captions (Inclusive)")
vq_parser.add_argument("--end",   type = int, default = None, help = "End index of captions (Exclusive)")
vq_parser.add_argument("-m",         "--clip_model", type=str, help="CLIP model (e.g. ViT-B/32, ViT-B/16)", default='RN50', dest='clip_model')
vq_parser.add_argument("-pre",       "--pretrained", default = True, action = "store_true", help = "Use the OpenAI pretrained models")
vq_parser.add_argument("-conf",      "--vqgan_config", type=str, help="VQGAN config", default=f'src/generation/VQGANCLIP/checkpoints/vqgan_imagenet_f16_16384.yaml', dest='vqgan_config')
vq_parser.add_argument("-ckpt",      "--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=f'src/generation/VQGANCLIP/checkpoints/vqgan_imagenet_f16_16384.ckpt', dest='vqgan_checkpoint')
vq_parser.add_argument("-clipckpt",  "--clip_checkpoint", default = None, type = str, help = "Path to pretrained CLIP", dest = 'clip_checkpoint')

vq_parser.add_argument("-i",    "--iterations", type=int, help="Number of iterations", default=200, dest='max_iterations')
vq_parser.add_argument("-se",   "--save_every", type=int, help="Save image iterations", default=50, dest='display_freq')
vq_parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height) (default: %(default)s)", default=[default_image_size,default_image_size], dest='size')
vq_parser.add_argument("-lr",   "--learning_rate", type=float, help="Learning rate", default=0.1, dest='step_size')
vq_parser.add_argument("-cutm", "--cut_method", type=str, help="Cut method", choices=['original','updated','nrupdated','updatedpooling','latest'], default='latest', dest='cut_method')
vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest='cutn')
vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')
vq_parser.add_argument("-opt",  "--optimiser", type=str, help="Optimiser", choices=['Adam','AdamW','Adagrad','Adamax','DiffGrad','AdamP','RAdam','RMSprop'], default='Adam', dest='optimiser')
vq_parser.add_argument("-cpe",  "--change_prompt_every", type=int, help="Prompt change frequency", default=0, dest='prompt_frequency')
vq_parser.add_argument("-d",    "--deterministic", action='store_true', help="Enable cudnn.deterministic?", dest='cudnn_determinism')
vq_parser.add_argument("-aug",  "--augments", nargs='+', action='append', type=str, choices=['Ji','Sh','Gn','Pe','Ro','Af','Et','Ts','Cr','Er','Re'], help="Enabled augments (latest vut method only)", default=[], dest='augments')
vq_parser.add_argument("-cd",   "--cuda_device", type=str, help="Cuda device to use", default="cuda:0", dest='cuda_device')


# Execute the parse_args() method
args = vq_parser.parse_args()

if args.cudnn_determinism:
    torch.backends.cudnn.deterministic = True

if not args.augments:
    args.augments = [['Af', 'Pe', 'Ji', 'Er']]

if not args.cuda_device == 'cpu' and not torch.cuda.is_available():
    args.cuda_device = 'cpu'
    print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
    print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")

device = torch.device(args.cuda_device)
gumbel = False

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

# NR: Testing with different intital images
def random_noise_image(w,h):
    random_image = Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    return random_image


# create initial gradient image
def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result

    
def random_gradient_image(w,h):
    array = gradient_3d(w, h, (0, 0, np.random.randint(0,255)), (np.random.randint(1,255), np.random.randint(2,255), np.random.randint(3,128)), (True, False, False))
    random_image = Image.fromarray(np.uint8(array))
    return random_image


# Used in older MakeCutouts
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply

def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)

#NR: Split prompts and weights
def split_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow # not used with pooling
        
        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            if item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True)) # border, reflection, zeros
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
                
        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        # self.noise_fac = False

        # Uncomment if you like seeing the list ;)
        # print(augment_list)
        
        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []
        
        for _ in range(self.cutn):            
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An updated version with Kornia augments and pooling (where my version started):
class MakeCutoutsPoolingUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow # Not used with pooling

        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),            
        )
        
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An Nerdy updated version with selectable Kornia augments, but no pooling:
class MakeCutoutsNRUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        
        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            if item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=30, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True)) # border, reflection, zeros
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
                
        self.augs = nn.Sequential(*augment_list)


    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An updated version with Kornia augments, but no pooling:
class MakeCutoutsUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),)
        self.noise_fac = 0.1


    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# This is the original version (No pooling)
class MakeCutoutsOrig(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)

class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

cut_size = 224 # hard-coded

# Cutout class options:
# 'latest','original','updated' or 'updatedpooling'
if args.cut_method == 'latest':
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
elif args.cut_method == 'original':
    make_cutouts = MakeCutoutsOrig(cut_size, args.cutn, cut_pow=args.cut_pow)
elif args.cut_method == 'updated':
    make_cutouts = MakeCutoutsUpdate(cut_size, args.cutn, cut_pow=args.cut_pow)
elif args.cut_method == 'nrupdated':
    make_cutouts = MakeCutoutsNRUpdate(cut_size, args.cutn, cut_pow=args.cut_pow)
else:
    make_cutouts = MakeCutoutsPoolingUpdate(cut_size, args.cutn, cut_pow=args.cut_pow)    

config_path = args.vqgan_config
checkpoint_path = args.vqgan_checkpoint

def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])

model = load_vqgan_model(config_path, checkpoint_path)
perceptor, processor = load(args.clip_model, pretrained = args.pretrained)
perceptor = perceptor.eval().requires_grad_(False)

f = 2**(model.decoder.num_resolutions - 1)
toksX, toksY = args.size[0] // f, args.size[1] // f
sideX, sideY = toksX * f, toksY * f

class Generator(nn.Module):

    def __init__(self, device = None):
        super(Generator, self).__init__()
        self.device = device
        self.model = model.to(device = self.device)
        self.perceptor = perceptor.to(device = self.device)

        if args.clip_checkpoint is not None:
            if os.path.isfile(args.clip_checkpoint):
                clip_ckpt = torch.load(args.clip_checkpoint, map_location = self.device)
                state_dict = clip_ckpt['state_dict']
                if(next(iter(state_dict.items()))[0].startswith("module")):
                    state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
                self.perceptor.load_state_dict(state_dict)
                print('checkpoint loaded!')
            else:
                print('no checkpoint found!')

        if gumbel:
            self.e_dim = 256
            self.n_toks = self.model.quantize.n_embed
            self.z_min = self.model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            self.z_max = self.model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            self.e_dim = self.model.quantize.e_dim
            self.n_toks = self.model.quantize.n_e
            self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
        
        self.z = self.get_z()
        self.z.requires_grad_(True)
        self.opt = self.get_opt(args.optimiser, args.step_size)


    def get_z(self):
        one_hot = F.one_hot(torch.randint(self.n_toks, [toksY * toksX], device=self.device), self.n_toks).float()
        if gumbel:
            z = one_hot @ self.model.quantize.embed.weight
        else:
            z = one_hot @ self.model.quantize.embedding.weight

        z = z.view([-1, toksY, toksX, self.e_dim]).permute(0, 3, 1, 2) 
        return z

    def get_prompt_obj(self, prompt):
        txt, weight, stop = split_prompt(prompt)
        txt_out = processor.process_text([txt])
        txt_input_ids, txt_attention_mask = txt_out['input_ids'].to(device = self.device), txt_out['attention_mask'].to(device = self.device)
        embed = self.perceptor.get_text_features(txt_input_ids, txt_attention_mask)
        return Prompt(embed, weight, stop).to(device = self.device)

    # Set the optimiser
    def get_opt(self, opt_name, opt_lr):
        if opt_name == "Adam":
            opt = optim.Adam([self.z], lr=opt_lr)   # LR=0.1 (Default)
        elif opt_name == "AdamW":
            opt = optim.AdamW([self.z], lr=opt_lr)  
        elif opt_name == "Adagrad":
            opt = optim.Adagrad([self.z], lr=opt_lr)    
        elif opt_name == "Adamax":
            opt = optim.Adamax([self.z], lr=opt_lr) 
        elif opt_name == "DiffGrad":
            opt = DiffGrad([self.z], lr=opt_lr, eps=1e-9, weight_decay=1e-9) # NR: Playing for reasons
        elif opt_name == "AdamP":
            opt = AdamP([self.z], lr=opt_lr)            
        elif opt_name == "RAdam":
            opt = optim.RAdam([self.z], lr=opt_lr)          
        elif opt_name == "RMSprop":
            opt = optim.RMSprop([self.z], lr=opt_lr)
        else:
            print("Unknown optimiser. Are choices broken?")
            opt = optim.Adam([self.z], lr=opt_lr)
        return opt

    def synth(self, z):
        if gumbel:
            z_q = vector_quantize(self.z.movedim(1, 3), self.model.quantize.embed.weight).movedim(3, 1)
        else:
            z_q = vector_quantize(self.z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

    @torch.inference_mode()
    def checkin(self, i, losses, prompt, output):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        out = self.synth(self.z)
        info = PngImagePlugin.PngInfo()
        info.add_text('comment', f'{prompt}')
        TF.to_pil_image(out[0].cpu()).save(f'{output}.png', pnginfo=info)   
    
    def ascend_txt(self, prompt_obj):
        out = self.synth(self.z)
        iii = self.perceptor.get_image_features(normalize(make_cutouts(out))).float()
        ls = prompt_obj(iii)
        return ls
    
    def train(self, i, prompt, output):

        self.opt.zero_grad(set_to_none=True)
        self.prompt_obj = self.get_prompt_obj(prompt)
        loss = self.ascend_txt(self.prompt_obj)
        
        if (i+1) % args.max_iterations == 0:
            self.checkin(i+1, [loss], prompt, output)
        
        loss.backward()
        self.opt.step()
        
        #with torch.no_grad():
        with torch.inference_mode():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))
    
    def forward(self, caption, fname):

        i = 0
        try:
            while i < args.max_iterations:
                self.train(i, caption, fname)
                i += 1
        except KeyboardInterrupt:
            pass
        
def get_generations(captions, args, queue):
    
    if args.device == 'cuda':
        rank = int(mp.current_process().name.replace('Process-', '')) - 1
        args.device = f'cuda:{rank}'
    
    for index, caption in captions:
        try:
            generator = Generator(args.device).to(args.device)
            fname = os.path.join(args.output_dir, f'{rank}_{index}.png')
            generator(caption, fname)
            queue.put([caption, os.path.join('images', f'{rank}_{index}.png')])
        except:
            queue.put(None)
    queue.put("Done")

def background(queue, bar, writer, num_processes):

    done = 0
    while 1:
        msg = queue.get()
        if msg is not None:
            if msg == 'Done':
                done += 1
            else:
                writer.writerow(msg)
                bar.update()
        else:
            bar.update()
        if done == num_processes:
            break

if __name__ == '__main__':

    if args.input_file.endswith('tsv'):
        args.separator = '\t'

    df = pd.read_csv(args.input_file, usecols = [args.caption_key], sep = args.separator)
    df = df.drop_duplicates(subset = [args.caption_key]).reset_index(drop = True)

    if args.start is None: args.start = 0
    if args.end is None: args.end = len(df)

    df = df[args.start : args.end]
    captions = set(df[args.caption_key])

    args.output_dir = f'{args.output_dir}-{args.start}-{args.end}'
    output_dir = os.path.join(args.output_dir, 'images/')
    output_file = os.path.join(args.output_dir, 'generated.csv')

    args.output_dir = output_dir
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
        args.device = "cpu"
        num_processes = 1
        print("Using cpu device")
    elif(ngpus == 1):
        args.device = "cuda"
        num_processes = 1
        print("Using gpu device")
        mp.set_start_method("spawn", force = True)
    else:
        args.device = "cuda"
        num_processes = ngpus
        print(f"Using multi-gpu devices ({num_processes})")
        mp.set_start_method("spawn", force = True)
    
    bar = tqdm(total = len(captions))

    queue = mp.Queue()
    captions = np.array_split(captions, num_processes)
    processes = [mp.Process(target = get_generations, args = (captions[i], args, queue)) for i in range(num_processes)]

    for process in processes:
        process.daemon = True
        process.start()

    background(queue, bar, writer, num_processes)

    for process in processes:
        process.join()

    print('Generation complete!')