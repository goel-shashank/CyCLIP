# https://github.com/moein-shariatnia/OpenAI-CLIP

import torch
import numpy as np
import albumentations as A
from transformers import DistilBertTokenizer

from . import config
from .model import CLIPModel

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if(p.grad):
            p.grad.data = p.grad.data.float()

class Processor:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.text_encoder_model)
        self.transform = A.Compose([A.Resize(config.image_size, config.image_size, always_apply = True), A.Normalize(max_pixel_value = 255.0, always_apply = True)])

    def process_text(self, text):
        if(isinstance(text, str)):
            text = [text]
        output = self.tokenizer(text, padding = True, truncation = True, max_length = config.max_sequence_length)
        return {"input_ids": torch.tensor(output["input_ids"]), "attention_mask": torch.tensor(output["attention_mask"])}

    def process_image(self, image):
        image = self.transform(image = np.array(image.convert("RGB")))["image"]
        return torch.tensor(image).permute(2, 0, 1).float()

def load(pretrained = False):
    model = CLIPModel(pretrained = pretrained)
    convert_models_to_fp32(model)
    processor = Processor()
    return model, processor