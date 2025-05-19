import os
import math

from tqdm import tqdm

import torch
import torch.nn.functional as F
import model

import dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

# configurations
batch_size = 64
max_epochs = 50000
start_epoch = 1

eval_interval = 1000
eval_iters = 200

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_iters = 2000
lr_decay_iters = max_epochs * 0.95

best_test_loss = float("inf")
always_save_checkpoint = False

resume = False
checkpoint_file = "accentator_checkpoint.pt"
output_dir = "../output"
checkpoint_path = os.path.join(output_dir, checkpoint_file)

torch.set_float32_matmul_precision("high")

model_config = model.Config(
    block_size=256,
    n_head=8,
    n_layer=8,
    n_embedding=512,
    dropout=0.1,
)

# load data
data = dataloader.Dataloader(
    path="../data",
    block_size=model_config.block_size,
)

model_config.vocab_size = data.get_vocab_size()

m = model.Accentator(model_config).to(device)

if resume:
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint["config"]
    m = model.Accentator(model_config).to(device)
    m.load_state_dict(checkpoint["model"])
    m.eval()

    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("Initializing model from scratch")

raw_m = m
m = torch.compile(m)

print(f"Number of parameters: {sum(p.numel() for p in m.parameters()):,}") # total params
optimizer = torch.optim.AdamW(m.parameters(), lr=max_lr)
if resume:
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Optimizer state loaded")

accentable = data.tokenizer.encode("aeiouyd")

@torch.no_grad()
def estimate_loss():
    out_losses = {}
    out_accuracies = {}
    out_true_accuracies = {}
    m.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        true_accuracies = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = data.get_batch(split, batch_size)
            logits, loss = m(xb, yb)
            losses[k] = loss.item()

            xb, yb = xb.view(-1), yb.view(-1)

            correct = (logits.view(-1, logits.size(-1)).argmax(dim=-1) == yb).float()
            accuracies[k] = correct.mean()
            
            accentable_mask = torch.zeros_like(correct, dtype=torch.bool)
            for a in accentable:
                accentable_mask |= (xb == a)
            
            accentable_count = accentable_mask.sum().item()
            true_correct = (correct * accentable_mask).sum().item()
            true_accuracies[k] = true_correct / accentable_count if accentable_count > 0 else 0

        out_losses[split] = losses.mean()
        out_accuracies[split] = accuracies.mean()
        out_true_accuracies[split] = true_accuracies.mean()

    m.train()

    return out_losses, out_accuracies, out_true_accuracies

def get_lr(it):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# training loop
for epoch in tqdm(range(start_epoch, max_epochs + 1)):
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    xb, yb = data.get_batch("train", batch_size)
    logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optimizer.step()

    if epoch % eval_interval == 0 or epoch == max_epochs - 1:
        print(f"Epoch {epoch}/{max_epochs}")
        losses, accuracies, true_accuracies = estimate_loss()

        print(f"Train loss: {losses['train']:.4f}, accuracy: {accuracies['train']:.4f}, true accuracy: {true_accuracies['train']:.4f}")
        print(f"Test loss: {losses['test']:.4f}, accuracy: {accuracies['test']:.4f}, true accuracy: {true_accuracies['test']:.4f}")

        if losses['test'] < best_test_loss or always_save_checkpoint:
            best_test_loss = losses['test']

            checkpoint = {
                'model': raw_m.state_dict(),
                'tokenizer': data.tokenizer,
                'config': model_config,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
            }

            print(f"Saving model to {checkpoint_path}")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)