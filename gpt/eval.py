import os

from tqdm import tqdm

import torch
import sacrebleu

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