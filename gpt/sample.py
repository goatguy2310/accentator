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

# Generate text
x, y = data.get_batch("test", 1)
print("Inpt:", data.tokenizer.decode(x[0].cpu().numpy()))
print("Targ:", data.tokenizer.decode(y[0].cpu().numpy()))

predicted = m.generate(x)
print("Pred:", data.tokenizer.decode(predicted[0].cpu().numpy()))

accentable = data.tokenizer.encode("aeiouyd")

correct, true_correct = 0, 0
accentable_count = 0
for i in range(len(predicted[0])):
    if predicted[0][i] == y[0][i]:
        correct += 1

    if x[0][i] in accentable:
        accentable_count += 1
        if predicted[0][i] == y[0][i]:
            true_correct += 1

print(f"Accuracy: {correct / len(predicted[0])} ({correct} / {len(predicted[0])})")
print(f"Accentable accuracy: {true_correct / accentable_count} ({true_correct} / {accentable_count})")