import torch
import torch.nn as nn

class GlobModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
