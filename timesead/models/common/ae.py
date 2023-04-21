from typing import Union, Tuple

import torch


class AE(torch.nn.Module):
    """
    Simple AE Implementation
    """
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, return_latent: bool = False):
        super(AE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.return_latent = return_latent

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z = self.encoder(x)
        x_pred = self.decoder(z)

        if self.return_latent:
            return x_pred, z

        return x_pred