import os

import matplotlib.pyplot as plt
import torch

import model
import dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

output_dir = "../output"
checkpoint_file = "accentator_checkpoint.pt"

checkpoint_path = os.path.join(output_dir, checkpoint_file)

print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

history = checkpoint.get("history", {"loss": [], "accuracy": [], "true_accuracy": []})

# Plotting the training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(history["loss"], label="Loss", color="blue")
plt.title("Test Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history["accuracy"], label="Accuracy", color="green")
plt.title("Test Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history["true_accuracy"], label="True Accuracy", color="orange")
plt.title("True Accuracy")
plt.xlabel("Iterations")
plt.ylabel("True Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_history.png"))