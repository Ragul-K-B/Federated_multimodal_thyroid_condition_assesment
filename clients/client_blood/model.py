import torch
import torch.nn as nn

# -------- Encoder --------
class BloodEncoder(nn.Module):
    def __init__(self, input_dim=5, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)


# -------- AutoEncoder (UNCHANGED) --------
class BloodAutoEncoder(nn.Module):
    def __init__(self, input_dim=5, embedding_dim=128):
        super().__init__()

        self.encoder = BloodEncoder(input_dim, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# -------- Federated Client Model (NEW) --------
class BloodClientModel(nn.Module):
    """
    Used ONLY for federated learning
    Encoder is local
    Classifier is federated
    """
    def __init__(self, encoder, num_classes=3):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)
