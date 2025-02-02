#%%writefile train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TransformerLM
from dataset import SimpleTextDataset

def collate_fn(batch):
    # Regroupe et padde les séquences de la batch
    inputs, targets = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

def main():
    # Exemple de textes pour l'entraînement
    texts = [
        "bonjour je suis un modele de langage",
        "apprendre a coder un llm est passionnant",
        "ce projet vous aide a comprendre les transformers",
        "hello world"
    ]
    dataset = SimpleTextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Création du modèle
    vocab_size = len(dataset.vocab)
    model = TransformerLM(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2,
                          dim_feedforward=256, dropout=0.1, max_seq_length=50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore_index pour le padding
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)  # (batch, seq_length, vocab_size)
            outputs = outputs.reshape(-1, vocab_size)
            batch_targets = batch_targets.reshape(-1)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Sauvegarde du modèle entraîné
    torch.save(model.state_dict(), "transformer_llm.pth")
    # Sauvegarde du vocabulaire pour l'inférence et l'API
    with open("vocab.txt", "w") as f:
        for token in dataset.vocab:
            f.write(token + "\n")

if __name__ == "__main__":
    main()
