from torch import nn

class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN):
    This helps the model capture multi-scale features more effectively.
    The FPN allows information to flow both top-down and bottom-up, which can improve the model's ability to detect slums of various sizes.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, x_skip=None):
        x = self.conv1(x)
        if x_skip is not None:
            x = self.upsample(x)
            x = x + x_skip
        x = self.conv2(x)
        return x