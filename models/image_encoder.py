import math
import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor):
        x = self.layer_norm_1(x)
        x = x + self.attn(x, x, x, need_weights = False)[0]
        x = self.layer_norm_2(x)
        x = x + self.feed_forward(x)
        return x
        
class ImageEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_dim = kwargs["input_dim"]
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.patch_size = kwargs["patch_size"]
        self.embed_dim = kwargs["embed_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.num_heads = kwargs["num_heads"]
        self.num_layers = kwargs["num_layers"]
        self.embedding_dim = kwargs["embedding_dim"]

        self.conv = nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels, kernel_size = self.patch_size, stride = self.patch_size, bias = False)
        self.class_embedding = nn.Parameter(torch.randn(self.out_channels) / math.sqrt(self.out_channels))
        self.positional_embedding = nn.Parameter(torch.randn((self.input_dim // self.patch_size) ** 2 + 1, self.out_channels) / math.sqrt(self.out_channels))
        self.layer_norm_1 = nn.LayerNorm(self.out_channels)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim = self.embed_dim, hidden_dim = self.hidden_dim, num_heads = self.num_heads) for _ in range(self.num_layers)])
        self.layer_norm_2 = nn.LayerNorm(self.out_channels)
        self.projection = nn.Parameter(torch.empty(self.out_channels, self.embedding_dim))

        self.initialize_parameters()
    
    def initialize_parameters(self):
        nn.init.normal_(self.projection, std = math.sqrt(self.embed_dim))
        
    def forward(self, x: torch.Tensor):
        x = self.conv(x)  
        x = x.flatten(start_dim = 2).permute(0, 2, 1) 
        x = torch.cat([self.class_embedding + torch.zeros((x.shape[0], 1, x.shape[-1]), device = x.device), x], dim = 1) + self.positional_embedding
        x = self.layer_norm_1(x)
        x = self.transformer(x)
        x = self.layer_norm_2(x[:, 0, :])
        x = x @ self.projection
        return x