import flwr as fl
import torch
import torch.nn as nn
import os

from glob_model import GlobModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ThermalFLClient(fl.client.NumPyClient):
    def __init__(self, model, features, labels):
        self.model = model.to(DEVICE)
        self.features = features.to(DEVICE)
        self.labels = labels.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        os.makedirs("models", exist_ok=True)

    def get_parameters(self, config=None):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict)

        # ⬇️ SAVE DOWNLOADED GLOBAL MODEL
        torch.save(
            self.model.state_dict(),
            "models/latest_global.pth"
        )
        print("⬇️ Global model downloaded (thermal)")

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        self.optimizer.zero_grad()
        outputs = self.model(self.features)
        loss = self.criterion(outputs, self.labels)
        loss.backward()
        self.optimizer.step()

        return self.get_parameters(), len(self.labels), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.labels), {}
