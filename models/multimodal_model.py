import torch 
import torch.nn as nn
from typing import List

class MultiModalModel(nn.Module):
    def __init__(self, embedding_dim: int, models: List):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.models = models

    def forward(self, inputs: List):
        num_models = len(inputs)       
        batch_size = inputs[0].shape[0]    
        embedding_dim = self.embedding_dim

        embeddings = torch.empty((num_models, batch_size, embedding_dim))
        for index, (model, input) in enumerate(zip(self.models, inputs)):
            embedding = model(input)
            embedding = embedding / embedding.norm(dim = -1, keepdim = True)
            embeddings[index] = embedding
        
        return embeddings
        


