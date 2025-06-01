from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    block_size: int = 3
    vocab_size: int = 768
    hidden_dim: int = 256 

class BigramModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Use smaller embedding dimension first
        self.word_embeddings = nn.Embedding(config.vocab_size, config.vocab_size)
        # good initialization for better convergence
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        logits = self.word_embeddings(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def generate(self, x):
        logits, _ = self.forward(x)
        logits = logits.argmax(dim=-1)
        return logits