import os

import torch

import model
import dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

output_dir = "../output"
checkpoint_file = "accentator_checkpoint.pt"

checkpoint_path = os.path.join(output_dir, checkpoint_file)

print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

model_config = checkpoint["config"]
m = model.Accentator(model_config).to(device)
m.load_state_dict(checkpoint["model"])
m.eval()

print(f"Context length: {model_config.block_size}")
print(f"Number of parameters: {sum(p.numel() for p in m.parameters()):,}") # total params

data = dataloader.Dataloader(
    path="../data",
    block_size=model_config.block_size,
)

x = input("Enter a sentence: ")
x = data.tokenizer.encode(x)
x = torch.tensor([x]).to(device)

y = m.generate(x)
print(f"Restored version: {data.tokenizer.decode(y[0].cpu().numpy())}")