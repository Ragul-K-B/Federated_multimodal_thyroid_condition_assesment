import flwr as fl

from client import BloodFLClient
from model import BloodClientModel
from dataloader import BloodDataset
from torch.utils.data import DataLoader
import torch

from model import BloodEncoder# your existing encoder

encoder = BloodEncoder()
encoder.load_state_dict(torch.load("blood_encoder.pth"))

dataset = BloodDataset("blood_data.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = BloodClientModel(encoder)

client = BloodFLClient(model, loader)

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client
)
