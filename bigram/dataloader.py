import torch
import tokenizer

class Dataloader:
    def __init__(self, path, block_size):
        self.path = path
        self.block_size = block_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.data = dict()
        self.tokenizer = tokenizer.Tokenizer()

        files = ['train.txt', 'train_normalized.txt', 'test.txt', 'test_normalized.txt']
        for file in files:
            self.tokenizer.load(f"{path}/{file}")

        for file in files:
            with open(f"{path}/{file}", 'r', encoding='utf-8') as f:
                txt = f.read()
                self.data[file[:-4]] = self.tokenizer.encode(txt)

    def get_batch(self, split, batch_size=1):
        data = self.data[split]
        data_n = self.data[split + "_normalized"]

        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([torch.tensor(data_n[i:i + self.block_size]) for i in ix])
        y = torch.stack([torch.tensor(data[i:i + self.block_size]) for i in ix])
        x, y = x.to(self.device), y.to(self.device)

        return x, y
    
    def get_split(self, split):
        data = self.data[split]
        data_n = self.data[split + "_normalized"]

        x = torch.tensor(data_n).to(self.device)
        y = torch.tensor(data).to(self.device)

        return x, y
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size

