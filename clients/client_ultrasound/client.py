import flwr as fl
import torch

from shared.utils import get_parameters, set_parameters


class UltrasoundFLClient(fl.client.NumPyClient):
    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)

            out = self.model(x)
            loss = self.criterion(out, y)

            loss.backward()

        return self.get_parameters(), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        return 0.0, len(self.loader.dataset), {}
