#%%writefile api.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import TransformerLM
from inference import load_vocab, generate_text

# Définition de la classe pour la requête API
class Prompt(BaseModel):
    text: str

app = FastAPI()

# Chargement du vocabulaire
vocab, token2idx, idx2token = load_vocab("vocab.txt")
vocab_size = len(vocab)

# Création du modèle et chargement des poids entraînés
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerLM(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2,
                      dim_feedforward=256, dropout=0.1, max_seq_length=50)
model.load_state_dict(torch.load("transformer_llm.pth", map_location=device))
model.to(device)
model.eval()

@app.post("/generate")
def generate(prompt: Prompt):
    generated_text = generate_text(model, prompt.text, token2idx, idx2token, device)
    return {"generated_text": generated_text}
