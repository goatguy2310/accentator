from tqdm import tqdm

import torch
import torch.nn.functional as F
import model

import dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

# configurations
batch_size = 32
max_epochs = 10

eval_interval = 1
eval_iters = 200

learning_rate = 1e-3

model_config = model.Config(
    block_size=1,
)

# load data
data = dataloader.Dataloader(
    path="../data",
    block_size=model_config.block_size,
)

model_config.vocab_size = data.get_vocab_size()
print("Vocab size:", model_config.vocab_size)

m = model.BigramModel(model_config).to(device)
print(f"Number of parameters: {sum(p.numel() for p in m.parameters()):,}") # total params

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

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

# training loop
for epoch in tqdm(range(1, max_epochs + 1)):
    xb, yb = data.get_batch("train")
    logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if epoch % eval_interval == 0 or epoch == max_epochs - 1:
        print(f"Epoch {epoch}/{max_epochs}")
        losses, accuracies, true_accuracies = estimate_loss()

        print(f"Train loss: {losses['train']:.4f}, accuracy: {accuracies['train']:.4f}, true accuracy: {true_accuracies['train']:.4f}")
        print(f"Test loss: {losses['test']:.4f}, accuracy: {accuracies['test']:.4f}, true accuracy: {true_accuracies['test']:.4f}")