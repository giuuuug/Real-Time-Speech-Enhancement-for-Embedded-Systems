import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 3),
        stride=(1, 2),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, C, T, F]
        Return: [B, C, T, F]
        """
        x = F.pad(x, (0, 0, 1, 0))
        return self.activation(self.bn(self.conv(x)))
