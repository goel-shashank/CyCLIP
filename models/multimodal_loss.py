import torch 
import torch.nn as nn

class MultiModalLoss(nn.Module):
    def __init__(self, temperature: float = 1, alphas: torch.Tensor = None):
        super().__init__()
        self.alphas = alphas
        self.temperature = temperature

    def __call__(self, embeddings: torch.Tensor):
        num_modalities, batch_size, embedding_dim = embeddings.shape # N x B x D

        x = embeddings.permute(1, 0, 2).flatten(start_dim = 0, end_dim = 1) # N x B x D > B x N x D > BN x D
        y = embeddings.permute(2, 0, 1).flatten(start_dim = 1, end_dim = 2) # N x B x D > D x N x B > D x NB

        g = x @ y # BN x NB
        g = g.reshape(batch_size, num_modalities, num_modalities, batch_size).permute(0, 3, 1, 2) # B x B x N x N
        g = (g / self.temperature).exp() # B x B x N x N
        g = g / g.sum(dim = 1, keepdim = True) # B x B x N x N
        g = g.diagonal() # N x N x B
        g = g.mean(dim = -1) # N x N

        loss = torch.sum(self.alphas * g)
        return loss