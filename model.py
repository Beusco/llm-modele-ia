# %%writefile model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # Ajustement pour d_model impair
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: Tensor de shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, max_seq_length=50):
        super(TransformerLM, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src):
        """
        src: Tensor de shape (batch_size, seq_length)
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        src = src.transpose(0, 1)  # Transformer attend (seq_length, batch_size, d_model)
        output = self.transformer_encoder(src)
        output = output.transpose(0, 1)  # Retour à (batch_size, seq_length, d_model)
        logits = self.fc_out(output)
        return logits
