import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU())
        self.attention = SpatialAttention(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)