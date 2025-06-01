from tqdm import tqdm

import sacrebleu

import torch
import torch.nn.functional as F

import model
import dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

# configurations
batch_size = 64
max_epochs = 50000

eval_interval = 1000
eval_iters = 100

learning_rate = 5e-4

model_config = model.Config(
    block_size=128,
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
    # Add gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optimizer.step()

    if epoch % eval_interval == 0 or epoch == max_epochs - 1:
        print(f"Epoch {epoch}/{max_epochs}")
        losses, accuracies, true_accuracies = estimate_loss()

        print(f"Train loss: {losses['train']:.4f}, accuracy: {accuracies['train']:.4f}, true accuracy: {true_accuracies['train']:.4f}")
        print(f"Test loss: {losses['test']:.4f}, accuracy: {accuracies['test']:.4f}, true accuracy: {true_accuracies['test']:.4f}")

losses, accuracies, true_accuracies, word_accuracies, bleus = [], [], [], [], []
with torch.no_grad():
    print("Evaluating model...")
    for i in tqdm(range(200)):
        x, y = data.get_batch("test", 64)

        accentable = data.tokenizer.encode("aeiouyd")
        m.eval()

        logits, loss = m(x, y)
        loss = loss.item()
        losses.append(loss)

        x, y = x.view(-1), y.view(-1)

        correct = (logits.view(-1, logits.size(-1)).argmax(dim=-1) == y).float()
        accuracy = correct.mean()
        accuracies.append(accuracy.item())
        
        accentable_mask = torch.zeros_like(correct, dtype=torch.bool)
        for a in accentable:
            accentable_mask |= (x == a)
        
        accentable_count = accentable_mask.sum().item()
        true_correct = (correct * accentable_mask).sum().item()
        true_accuracy = true_correct / accentable_count if accentable_count > 0 else 0
        true_accuracies.append(true_accuracy)

        decoded_y = data.tokenizer.decode(y.cpu().numpy())
        decoded_pred = data.tokenizer.decode(logits.argmax(dim=-1).view(-1).cpu().numpy())
        correct_words = sum([1 if w == yb else 0 for w, yb in zip(decoded_pred.split(), decoded_y.split())])
        word_accuracy = correct_words / len(decoded_y.split()) if len(decoded_y.split()) > 0 else 0
        word_accuracies.append(word_accuracy)

        # Calculate BLEU score
        bleu = sacrebleu.corpus_bleu(decoded_pred.split(), [decoded_y.split()])
        bleus.append(bleu.score)


def mean(lst):
    return sum(lst) / len(lst) if lst else 0

print(f"Loss: {mean(losses)}")
print(f"Accuracy: {mean(accuracies)}")
print(f"Accentable accuracy: {mean(true_accuracies)}")
print(f"Word accuracy: {mean(word_accuracies)}")
print(f"BLEU score: {mean(bleus)}")