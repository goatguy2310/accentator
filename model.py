import math

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    block_size: int = 512
    vocab_size: int = 768
    n_head: int = 6
    n_layer: int = 6
    n_embedding: int = 256
    dropout: float = 0.1

class CausalAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # key, query, value projection
        self.attention = nn.Linear(config.n_embedding, config.n_embedding * 3)
        # output projection
        self.output_linear = nn.Linear(config.n_embedding, config.n_embedding)

        # layer normalization
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embedding = config.n_embedding
        self.n_dropout = config.dropout

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # batch size, length of sequence and embedding size
        batch_size, seq_len, emb_size = x.size()

        # extract key, query and value then reshape
        q, k, v = self.attention(x).split(self.n_embedding, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, emb_size // self.n_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, emb_size // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, emb_size // self.n_head).transpose(1, 2)

        # attention calculation
        att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout1(att)
        out = att @ v

        # merge heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_size)

        # output projection
        out = self.dropout2(self.output_linear(out))
        return out
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.n_embedding, config.n_embedding * 4)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(config.n_embedding * 4, config.n_embedding)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_embedding, eps=1e-5)

    def forward(self, x):
        return self.layer_norm(x)
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = LayerNorm(config)
        self.attention = CausalAttentionLayer(config)
        self.layer_norm2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # residual connection
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x
    
class Accentator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # the transformer
        self.wte = nn.Embedding(config.vocab_size, config.n_embedding)
        self.wpe = nn.Embedding(config.block_size, config.n_embedding)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.layer_norm = LayerNorm(config.n_embedding)

        # the final linear layer
        self.linear_head = nn.Linear(config.n_embedding, config.vocab_size, bias=False)

        # weight tying
        self.wte.weight = self.linear_head.weight

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        _, seq_len = x.size()

        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        
        token_embeddings = self.wte(x)
        position_embeddings = self.wpe(pos)
        x = self.dropout(token_embeddings + position_embeddings)
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)

        out = self.linear_head(x[:, -1, :])
        return out
    
    @torch.no_grad()
    def generate(self, x, max_length):
        for _ in range(max_length):
            x = x if x.size(1) <= self.config.block_size else x[:, -self.config.block_size:]
            out = self(x)
            _, next_token = out[:, -1, :].topk(1)
            next_token = next_token[:, -1].unsqueeze(1)

            x = torch.cat((x, next_token), dim=1)
            if x.size(1) >= self.config.block_size:
                break
        return x