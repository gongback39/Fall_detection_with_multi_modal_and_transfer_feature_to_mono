import torch
import torch.nn as nn

class R2Plus1D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super(R2Plus1D_Block, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.spatial_conv = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.temporal_conv = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            padding=(1, 0, 0),
            bias=False
        )

        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.temporal_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class R2Plus1DNet(nn.Module):
    def __init__(self, num_classes=400):
        super(R2Plus1DNet, self).__init__()
        self.layer1 = R2Plus1D_Block(3, 64, stride=1)
        self.layer2 = R2Plus1D_Block(64, 128, stride=2)
        self.layer3 = R2Plus1D_Block(128, 128, stride=2)
        self.layer4 = R2Plus1D_Block(128, 64, stride=2)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x