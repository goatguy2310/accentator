import torch
import torch.nn.functional as F
import model

device = "cuda" if torch.cuda.is_available() else "cpu"

# configurations
batch_size = 32
max_epochs = 10
learning_rate = 1e-3
eval_interval = 1

model_config = model.Config(
    block_size=1,
    vocab_size=768,
)

m = model.BigramModel(model_config).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# training loop
for epoch in range(max_epochs):
    xb, yb = get_batch(train_data, batch_size)
    logits = m(xb)
    loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), yb.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if epoch % eval_interval == 0 or epoch == max_epochs - 1:
        print(f"Epoch {epoch}/{max_epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Eval loss: {eval_loss:.4f}")