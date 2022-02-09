# Code ported from https://github.com/openai/CLIP

import os
import torch
import urllib
import hashlib
import warnings
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop

from utils import config
from .model import build
from .tokenizer import SimpleTokenizer as Tokenizer

models = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def download(url, root = os.path.expanduser(f"{config.root}/.cache/openai")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

class Processor:
    def __init__(self, model):
        self.tokenizer = Tokenizer()
        self.sot_token = self.tokenizer.encoder["<start_of_text>"]
        self.eot_token = self.tokenizer.encoder["<end_of_text>"]
        self.context_length = 77

        self.transform = Compose([Resize(model.visual.input_resolution, interpolation = Image.BICUBIC), CenterCrop(model.visual.input_resolution), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    def process_text(self, texts):
        if(isinstance(texts, str)):
            texts = [texts]

        result = torch.zeros(len(texts), self.context_length, dtype = torch.long)

        for i, text in enumerate(texts):
            tokens = [self.sot_token] + self.tokenizer.encode(text) + [self.eot_token]
            if(len(tokens) > self.context_length):
                tokens = tokens[:self.context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)

        return {"input_ids": result, "attention_mask": torch.empty((len(result),))}

    def process_image(self, image):
        return self.transform(image.convert("RGB"))

def load(name, pretrained = False):
    if(name in models):
        model_path = download(models[name])
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list(models.keys())}")

    model = torch.jit.load(model_path, map_location= "cpu").eval()

    try:
        model = build(model.state_dict(), pretrained = pretrained)
    except KeyError:
        state_dict = {key["module.":]: value for key, value in state_dict["state_dict"].items()}
        model = build(state_dict, pretrained = pretrained)

    convert_models_to_fp32(model)
    processor = Processor(model)

    return model, processor