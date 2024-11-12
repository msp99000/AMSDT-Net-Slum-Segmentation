from torch import nn
import torch

class MultiScaleFeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, dilation=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=2, dilation=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=4, dilation=4)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)