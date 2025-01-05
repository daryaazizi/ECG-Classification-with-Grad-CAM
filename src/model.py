from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels=32,
            kernel_size=5,
            padding='same'
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=5,
            padding='same'
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = self.conv2(x1)
        x1 = F.relu(x1 + x)
        return self.maxpool(x1)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            padding='same'
        )
        
        residual_blocks = [ResidualBlock(32, 32) for _ in range(5)]
        self.convnet = nn.Sequential(*residual_blocks)
        
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=32*4, out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=5)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.convnet(x1)
        z = self.classifier(x2)
        return z
    