import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, input_length):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)  # 初始卷积
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 残差块，设置几个不同深度的模块
        self.layer1 = self._make_layer(64, 64, kernel_size=3, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, kernel_size=3, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, kernel_size=3, blocks=2, stride=2)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出归一化到 [0,1]
        )

    def _make_layer(self, in_channels, out_channels, kernel_size, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, kernel_size, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, input_length)
        x = x.unsqueeze(1)  # -> (batch_size, 1, input_length)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


class CNN(nn.Module):
    def __init__(self, input_length):
        """
        input_length: 特征维度长度
        """
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
            nn.Sigmoid()  # 输出归一化到 [0,1]
        )

    def forward(self, x):
        # x shape: (batch_size, input_length)
        x = x.unsqueeze(1)  # -> (batch_size, 1, input_length)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


class simpleCNN(nn.Module):
    def __init__(self, input_length):
        """
        input_length: 特征维度长度
        """
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
            nn.Sigmoid()  # 输出归一化到 [0,1]
        )

    def forward(self, x):
        # x shape: (batch_size, input_length)
        x = x.unsqueeze(1)  # -> (batch_size, 1, input_length)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


# 示例实例化
if __name__ == "__main__":
    model = Resnet(input_length=500)
    sample_input = torch.randn(8, 500)  # batch_size=8
    output = model(sample_input)
    print(output.shape)  # 应输出 torch.Size([8])
