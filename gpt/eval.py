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

print(f"Number of parameters: {sum(p.numel() for p in m.parameters()):,}") # total params

data = dataloader.Dataloader(
    path="../data",
    block_size=model_config.block_size,
)

x, y = data.get_batch("test", 64)
with torch.no_grad():
    print("Evaluating model...")

    accentable = data.tokenizer.encode("aeiouyd")
    m.eval()

    logits, loss = m(x, y)
    loss = loss.item()

    x, y = x.view(-1), y.view(-1)

    correct = (logits.view(-1, logits.size(-1)).argmax(dim=-1) == y).float()
    accuracy = correct.mean()
    
    accentable_mask = torch.zeros_like(correct, dtype=torch.bool)
    for a in accentable:
        accentable_mask |= (x == a)
    
    accentable_count = accentable_mask.sum().item()
    true_correct = (correct * accentable_mask).sum().item()
    true_accuracy = true_correct / accentable_count if accentable_count > 0 else 0

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy} ({correct.sum().item()} / {len(correct)})")
print(f"Accentable accuracy: {true_accuracy} ({true_correct} / {accentable_count})")