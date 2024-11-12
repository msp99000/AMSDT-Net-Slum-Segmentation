from torch import nn

class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * out_channels * kernel_size * kernel_size, 1)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.weight_gen(x).view(b, c, self.kernel_size, self.kernel_size)
        return F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=b)