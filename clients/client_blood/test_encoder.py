import torch
from model import BloodEncoder
from dataloader import BloodDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = BloodDataset("R:\\final_proj\\thyroid\\clients\\client_blood\\blood_data.csv")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model
model = BloodEncoder(input_dim=5, embedding_dim=128)
model.eval()

# Test forward pass
with torch.no_grad():
    for batch in loader:
        embeddings = model(batch)
        print("Input shape:", batch.shape)
        print("Embedding shape:", embeddings.shape)
        break
