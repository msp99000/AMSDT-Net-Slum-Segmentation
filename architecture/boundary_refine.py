from torch import nn
import torch.nn.functional as F

class BoundaryRefinementModule(nn.Module):
    """
    It uses residual connections to help refine the segmentation boundaries.
    It applies two convolutional layers with a residual connection, which allows the module to learn small refinements to the segmentation map.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)