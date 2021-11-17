import math
import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, attn_mask: torch.Tensor):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_mask = attn_mask

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
        x = x + self.attn(x, x, x, attn_mask = self.attn_mask.to(x.device), need_weights = False)[0]
        x = self.layer_norm_2(x)
        x = x + self.feed_forward(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.vocab_size = kwargs["vocab_size"]
        self.context_size = kwargs["context_size"]
        self.embed_dim = kwargs["embed_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.num_heads = kwargs["num_heads"]
        self.num_layers = kwargs["num_layers"]
        self.embedding_dim = kwargs["embedding_dim"]

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_size, self.embed_dim))
        self.attn_mask = torch.empty(self.context_size, self.context_size).fill_(float("-inf")).triu_(1)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim = self.embed_dim, hidden_dim = self.hidden_dim, num_heads = self.num_heads, attn_mask = self.attn_mask) for _ in range(self.num_layers)])
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.projection = nn.Parameter(torch.empty(self.embed_dim, self.embedding_dim))

        self.initialize_parameters()
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std = 0.02)
        nn.init.normal_(self.positional_embedding, std = 0.01)

        for attn_block in self.transformer.children():
            nn.init.normal_(attn_block.attn.in_proj_weight, std = 1 / math.sqrt(2 * self.num_layers * attn_block.embed_dim))
            nn.init.normal_(attn_block.attn.out_proj.weight, std = 1 / math.sqrt(attn_block.embed_dim))
            nn.init.normal_(attn_block.feed_forward[0].weight, std = 2 / math.sqrt(attn_block.embed_dim))
            nn.init.normal_(attn_block.feed_forward[2].weight, std = 1 / math.sqrt(attn_block.embed_dim))
        
        nn.init.normal_(self.projection, std = math.sqrt(self.embed_dim))

    def forward(self, text: torch.Tensor):
        x = self.token_embedding(text) + self.positional_embedding
        x = self.layer_norm(self.transformer(x))
        x = x[torch.arange(x.shape[0]), text.argmax(dim = -1)] 
        x = x @ self.projection
        return x