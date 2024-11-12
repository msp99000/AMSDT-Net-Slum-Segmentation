from torch import nn

class Input(nn.Module):
    """Single-modality module for input pass (RGB satellite imagery)"""
    def __init__(self, satellite_channels):
        super().__init__()
        self.conv = nn.Conv2d(satellite_channels, 64, kernel_size=3, padding=1)
    
    def forward(self, img):
        return self.conv(img)