import flwr as fl
import torch
import torch.nn as nn
from model import BloodClientModel
from dataloader import BloodDataset
from shared.utils import get_parameters, set_parameters
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BloodFLClient(fl.client.NumPyClient):
    def __init__(self, model, loader):
        self.model = model.to(DEVICE)
        self.loader = loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config=None):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for x, y in self.loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = self.model(x)
            loss = self.criterion(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.get_parameters(), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.loader.dataset), {}
