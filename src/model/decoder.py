import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 2),
        stride=(1, 2),
        output_padding=(0, 0),
        is_last=False,
    ) -> None:
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            output_padding=output_padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU() if not is_last else nn.ReLU()

    def forward(self, x, target_time=None):
        x = self.deconv(x)
        # Taglia la dimensione temporale per matchare il target
        if target_time is not None and x.shape[2] > target_time:
            x = x[:, :, :target_time, :]
        elif x.shape[2] > 1:
            x = x[:, :, :-1, :]
        return self.activation(self.bn(x))
