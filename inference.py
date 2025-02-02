#%%writefile inference.py
import torch
from model import TransformerLM

def load_vocab(vocab_file="vocab.txt"):
    with open(vocab_file, "r") as f:
        tokens = f.read().splitlines()
    token2idx = {token: idx for idx, token in enumerate(tokens)}
    idx2token = {idx: token for idx, token in enumerate(tokens)}
    return tokens, token2idx, idx2token

def generate_text(model, prompt, token2idx, idx2token, device, max_length=20):
    model.eval()
    tokens = prompt.split()
    # Convertir les tokens en indices, avec 0 comme index par défaut pour un token inconnu
    input_ids = [token2idx.get(token, 0) for token in tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_tensor)  # (1, seq_length, vocab_size)
            next_token_logits = outputs[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            if idx2token[next_token_id] == '<eos>':
                break
            generated.append(next_token_id)
            # Concaténer le nouveau token à la séquence d'entrée
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)
    
    generated_text = ' '.join([idx2token[idx] for idx in generated])
    return generated_text

def main():
    # Chargement du vocabulaire
    vocab, token2idx, idx2token = load_vocab("vocab.txt")
    vocab_size = len(vocab)
    
    # Création et chargement du modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2,
                          dim_feedforward=256, dropout=0.1, max_seq_length=50)
    model.load_state_dict(torch.load("transformer_llm.pth", map_location=device))
    model.to(device)
    
    # Exemple d'inférence
    prompt = "bonjour je suis"
    generated_text = generate_text(model, prompt, token2idx, idx2token, device)
    print("Prompt:", prompt)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
