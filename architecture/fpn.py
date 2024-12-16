import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Add a 1x1 convolution to align channels
        self.channel_projector = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, x_skip):
        # Process the input feature map x
        x = self.conv1(x)  # Downsampled feature map

        if x_skip is not None:
            # Align the spatial dimensions of x_skip with x
            x_skip_resized = F.interpolate(x_skip, size=x.shape[2:], mode="bilinear", align_corners=False)

            # Align the channels of x_skip with x using a 1x1 convolution
            x_skip_resized = self.channel_projector(x_skip_resized)

            # Add the skip connection
            x = x + x_skip_resized

        x = self.conv2(x)
        return x




# class FPN(nn.Module):
#     """
#     Feature Pyramid Network (FPN):
#     This helps the model capture multi-scale features more effectively.
#     The FPN allows information to flow both top-down and bottom-up, which can improve the model's ability to detect slums of various sizes.
#     """
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

#     def forward(self, x, x_skip=None):
#         x = self.conv1(x)
#         if x_skip is not None:
#             x = self.upsample(x)
#             x = x + x_skip
#         x = self.conv2(x)
#         return x