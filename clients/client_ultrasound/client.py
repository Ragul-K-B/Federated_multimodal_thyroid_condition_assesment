import flwr as fl
import torch
import torch.nn as nn
import os
from glob_model import GlobModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class UltrasoundFLClient(fl.client.NumPyClient):
    def __init__(self, features, labels):
        # 🔥 FEATURES ARE ALREADY ENCODED (N,128)
        self.features = features.to(DEVICE)
        self.labels = labels.to(DEVICE)

        # ===== GLOBAL MODEL =====
        self.model = GlobModel(input_dim=128, num_classes=3).to(DEVICE)

        # ===== CLASS COUNTS =====
        normal = 164
        benign = 292
        malignant = 280
        total = normal + benign + malignant

        weights = torch.tensor(
            [total / normal, total / benign, total / malignant],
            dtype=torch.float
        ).to(DEVICE)

        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        os.makedirs("models", exist_ok=True)

    # ================= PARAMETERS =================
    def get_parameters(self, config=None):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict)
        torch.save(self.model.state_dict(), "models/latest_global.pth")

    # ================= TRAIN =================
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        LOCAL_EPOCHS = 5

        for _ in range(LOCAL_EPOCHS):
            self.optimizer.zero_grad()
            outputs = self.model(self.features)
            loss = self.criterion(outputs, self.labels)
            loss.backward()
            self.optimizer.step()

        return self.get_parameters(), len(self.labels), {}

    # ================= EVALUATE =================
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.labels), {}
