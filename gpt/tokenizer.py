import string

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

class SyllableTokenizer:
    def __init__(self, max_size):
        self.vocab = set()
        self.vocab_size = 0
        self.max_size = max_size

        self.stoi = None
        self.itos = [""] * max_size

    def load(self, file_path):
        txt = ""
        with open(file_path, 'r', encoding='utf-8') as f:
            txt = f.read()
        
        self.vocab.update(set(self.get_syllables(txt)))

        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        for i, ch in enumerate(self.vocab):
            if i < self.max_size:
                self.itos[i] = ch
            else:
                break


    def get_syllables(self, text):
        syllables = (''.join([ch if not ch in string.punctuation else ' ' for ch in text])).strip().split()
        return syllables

    def encode(self, text):
        tokens = []
        for syllable in self.get_syllables(text):
            if syllable in self.stoi:
                tokens.append(self.stoi[syllable])
        return tokens
    
    def decode(self, tokens):
        new_text = ""
        for t in tokens:
            new_text += self.itos[t] + ' '
        
        return new_text