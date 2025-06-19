import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1. Residual Block & ResNet
# -----------------------------

class ResidualBlock1D(nn.Module):
    """
    A 1D residual block with two convolutional layers and skip connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class Resnet(nn.Module):
    """
    1D ResNet for multi-class classification (default 4 outputs with softmax).
    """
    def __init__(self, input_length):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.block1 = ResidualBlock1D(64)
        self.block2 = ResidualBlock1D(64)
        self.block3 = ResidualBlock1D(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 4)  # Default: 4-class output

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_length)
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)  # (batch_size, 64)
        return F.softmax(self.fc(x), dim=1)


# -----------------------------
# 2. CNN (Deep) for Binary Classification
# -----------------------------

class CNN(nn.Module):
    """
    Deep 1D CNN for binary classification. Final output: sigmoid scalar in [0,1].
    """
    def __init__(self, input_length):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * input_length, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_length)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


# -----------------------------
# 3. Simple CNN for Binary Classification
# -----------------------------

class simpleCNN(nn.Module):
    """
    Simpler version of CNN with fewer layers for binary classification.
    """
    def __init__(self, input_length):
        super(simpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * input_length, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_length)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


# -----------------------------
# 4. Example Usage
# -----------------------------

if __name__ == "__main__":
    model = Resnet(input_length=500)
    sample_input = torch.randn(8, 500)  # batch size = 8, input length = 500
    output = model(sample_input)
    print("Output shape:", output.shape)  # Expected: (8, 4)
