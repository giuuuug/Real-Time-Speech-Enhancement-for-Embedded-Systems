import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(1, 2)) -> None:
        super().__init__()
        self.padding = (kernel_size[0] - 1, 0)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        x = F.pad(x, (0, 0, self.padding[0], 0))
        return self.activation(self.bn(self.conv(x)))
