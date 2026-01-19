import torch
import torch.nn as nn

# -------- Thermal Encoder (PRIVATE, NOT FEDERATED) --------
class ThermalEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self.encoder = nn.Linear(128 * 28 * 28, feature_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        features = self.encoder(x)
        return features


# -------- Federated Client Model --------
class ThermalClientModel(nn.Module):
    """
    Encoder stays local
    Classifier participates in federated learning
    """
    def __init__(self, encoder, num_classes=3):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)
