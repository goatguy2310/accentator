class Tokenizer:
    def __init__(self):
        self.vocab = ""
        self.vocab_size = 0
        self.stoi = None
        self.itos = None

    def load(self, file_path):
        txt = ""
        with open(file_path, 'r', encoding='utf-8') as f:
            txt = f.read()

        self.vocab = ''.join(sorted(list(set(self.vocab + txt))))
        self.vocab_size = len(self.vocab)

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]
    
    def decode(self, tokens):
        return ''.join([self.itos[token] for token in tokens])
    
    def reset(self):
        self.vocab = ""
        self.vocab_size = 0
        self.stoi = None
        self.itos = None