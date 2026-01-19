import torch.nn as nn

class UltrasoundEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
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

        # ONLY for supervised training
        self.classifier = nn.Linear(feature_dim, 3)  # normal / benign / malignant

    def forward(self, x, extract_features=False):
        x = self.cnn(x)
        x = self.flatten(x)
        features = self.encoder(x)

        if extract_features:
            return features

        return self.classifier(features)
