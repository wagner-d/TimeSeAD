import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, conv_layer, out_channels: int, activation, batch_norm: bool = False):
        super(ConvBlock, self).__init__()

        self.conv = conv_layer
        self.activation = activation
        self.norm = torch.nn.BatchNorm1d(out_channels) if batch_norm else torch.nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x, *args, **kwargs)))
