from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    block_size: int = 1
    vocab_size: int = 768

class BigramModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.vocab_size)

    def forward(self, x):
        out = self.word_embeddings(x)
        return out
    
    def generate(self, x, max_length):
        for _ in range(max_length):
            logits = self.forward(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

            if x.size(1) >= self.config.block_size:
                break
        return x