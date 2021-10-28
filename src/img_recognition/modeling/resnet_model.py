import torch
from torch import nn
import torch.nn.functional as F

from img_recognition.layers import ResNetBlock, ResNetChangeBlock

class ResNet(nn.Module):
    
    def __init__(self, n: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, padding=1)
        self.resblock1_list = nn.ModuleList([
            ResNetBlock(in_channels=16) for _ in range(n)
        ])
        self.resblock2_list = nn.ModuleList([
                ResNetChangeBlock(in_channels=16, num_filters=32)
            ] + [
            ResNetBlock(in_channels=32) for _ in range(n-1)
        ])
        self.resblock3_list = nn.ModuleList([
            ResNetChangeBlock(in_channels=32, num_filters=64)
        ] + [
            ResNetBlock(in_channels=64) for _ in range(n-1)
        ])
        self.linear1 = nn.Linear(in_features=64,
                                 out_features=num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        for layer1 in self.resblock1_list:
            x = layer1(x)
        for layer2 in self.resblock2_list:
            x = layer2(x)
        for layer3 in self.resblock3_list:
            x = layer3(x)
        x = torch.mean(x, dim=[2, 3])
        x = self.linear1(x)
        return x