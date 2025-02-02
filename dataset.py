#%%writefile dataset.py
import torch
from torch.utils.data import Dataset

class SimpleTextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.vocab = self.build_vocab(texts)
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
    
    def build_vocab(self, texts):
        tokens = set()
        for text in texts:
            tokens.update(text.split())
        tokens.add('<eos>')  # Token de fin de séquence
        return sorted(list(tokens))
    
    def encode(self, text):
        tokens = text.split() + ['<eos>']
        return [self.token2idx[token] for token in tokens]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded = self.encode(self.texts[idx])
        # Pour ce dataset, l'entrée et la cible sont identiques.
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(encoded, dtype=torch.long)
