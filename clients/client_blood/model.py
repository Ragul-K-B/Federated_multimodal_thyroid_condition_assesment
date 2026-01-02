import torch
import torch.nn as nn

class BloodEncoder(nn.Module):
    """
    Final deployment encoder.
    Used for:
    - Feature extraction
    - Federated learning input
    """

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
        """
        x: (batch_size, 5)
        returns: (batch_size, 128)
        """
        return self.net(x)
