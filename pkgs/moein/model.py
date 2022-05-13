import timm
import torch
import numpy as np
from torch import nn
from transformers import DistilBertModel, DistilBertConfig

from . import config

class ImageEncoder(nn.Module):
    def __init__(self, model = config.image_encoder_model, pretrained = True):
        super().__init__()
        
        self.model = timm.create_model(model, pretrained, num_classes = 0, global_pool = "avg")
        
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def forward(self, pixel_values):
        return self.model(pixel_values)

class TextEncoder(nn.Module):
    def __init__(self, model = config.text_encoder_model, pretrained = True):
        super().__init__()
        
        if(pretrained):
            self.model = DistilBertModel.from_pretrained(model)
        else:
            self.model = DistilBertModel(config = DistilBertConfig())
            
        for paramter in self.model.parameters():
            paramter.requires_grad = True

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state[:, 0, :]

class ProjectionHead(nn.Module):
    def __init__(self, input_embedding_dim, output_embedding_dim = config.output_embedding_dim, dropout = config.dropout):
        super().__init__()
        self.projection = nn.Linear(input_embedding_dim, output_embedding_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_embedding_dim, output_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(output_embedding_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        return self.ln(self.dropout(self.fc(self.gelu(projected))) + projected)
    
class CLIPOutput:
    def __init__(self, image_embeds, text_embeds):
        self.image_embeds = image_embeds
        self.text_embeds = text_embeds

class CLIPModel(nn.Module):
    def __init__(self, pretrained, temperature = config.temperature, image_embedding_dim = config.image_embedding_dim, text_embedding_dim = config.text_embedding_dim):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained = pretrained)
        self.text_encoder = TextEncoder(pretrained = pretrained)
        self.image_projection = ProjectionHead(input_embedding_dim = image_embedding_dim)
        self.text_projection = ProjectionHead(input_embedding_dim = text_embedding_dim)
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_ids, attention_mask, pixel_values):
        image_features = self.image_projection(self.image_encoder(pixel_values = pixel_values))
        text_features = self.text_projection(self.text_encoder(input_ids = input_ids, attention_mask = attention_mask))
   
        image_features = image_features / image_features.norm(dim = -1, keepdim = True)
        text_features = text_features / text_features.norm(dim = -1, keepdim = True)

        output = CLIPOutput(image_features, text_features)        
        return output
