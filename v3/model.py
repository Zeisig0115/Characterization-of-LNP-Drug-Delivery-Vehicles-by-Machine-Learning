import torch.nn as nn


class simpleCNN(nn.Module):
    def __init__(self, input_length):
        super(simpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm1d(out_channels)

        # 如果 in/out 通道不匹配，或者 stride>1，需要投影一下以保证维度能相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class DeepResNet1D(nn.Module):
    def __init__(self, input_length, num_blocks=[2, 2, 2, 2]):
        """
        num_blocks: 每个stage里要堆叠几个残差block
        """
        super(DeepResNet1D, self).__init__()

        # 第一层做一下通道转换
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm1d(16)
        self.relu  = nn.ReLU(inplace=True)

        # 接下来堆4个stage，每个stage若干残差block
        self.stage1 = self._make_stage(16, 16,  num_blocks[0], stride=1)
        self.stage2 = self._make_stage(16, 32,  num_blocks[1], stride=2)
        self.stage3 = self._make_stage(32, 64,  num_blocks[2], stride=2)
        self.stage4 = self._make_stage(64, 128, num_blocks[3], stride=2)

        # 池化+全连接
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.drop1 =  nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(128, 32)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(32, 1)

    def _make_stage(self, in_channels, out_channels, blocks, stride):
        """构造一个stage，包含若干个ResidualBlock"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # [B, length] -> [B, 1, length]
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # -> [B, 128]
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)   # -> [B, 1]
        return x




