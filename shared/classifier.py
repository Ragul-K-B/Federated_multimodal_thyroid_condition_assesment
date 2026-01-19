import torch.nn as nn

class ThyroidClassifier(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
