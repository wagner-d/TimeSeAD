from typing import Union, Callable, Tuple, List, Sequence

import torch

from ..layers import SameCausalZeroPad1d
from ...utils import torch_utils


class TCNResidualBlock(torch.nn.Module):
    def __init__(self, input_dim: int, dilation_rate: int, nb_filters: int, kernel_size: int,
                 padding: str, activation: Union[str, Callable] = 'relu',
                 dropout_rate: float = 0, use_batch_norm: bool = False, use_layer_norm: bool = False):
        """
        Defines the residual block for the WaveNet TCN. Input needs to be of shape (B, D, T).

        Args:
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
        """

        super().__init__()

        if padding == 'causal':
            self.conv1 = torch.nn.Sequential(
                SameCausalZeroPad1d(kernel_size, dilation=dilation_rate),
                torch.nn.Conv1d(input_dim, nb_filters, kernel_size=kernel_size, dilation=dilation_rate, padding=0)
            )
            self.conv2 = torch.nn.Sequential(
                SameCausalZeroPad1d(kernel_size, dilation=dilation_rate),
                torch.nn.Conv1d(nb_filters, nb_filters, kernel_size=kernel_size, dilation=dilation_rate, padding=0)
            )
        else:
            self.conv1 = torch.nn.Conv1d(input_dim, nb_filters, kernel_size=kernel_size, dilation=dilation_rate,
                                         padding=padding)
            self.conv2 = torch.nn.Conv1d(nb_filters, nb_filters, kernel_size=kernel_size, dilation=dilation_rate,
                                         padding=padding)

        if use_batch_norm:
            self.norm1 = torch.nn.BatchNorm1d(nb_filters)
            self.norm2 = torch.nn.BatchNorm1d(nb_filters)
        elif use_layer_norm:
            self.norm1 = torch.nn.LayerNorm(nb_filters)
            self.norm2 = torch.nn.LayerNorm(nb_filters)
        else:
            self.norm1 = torch.nn.Identity()
            self.norm2 = torch.nn.Identity()

        if isinstance(activation, str):
            activation = torch_utils.activations[activation]
        self.activation = activation

        # Dropout1d does not exist in PyTorch for some reason, but Dropout2d works with 1D data as well
        # and does exactly what we want.
        # See https://github.com/pytorch/pytorch/issues/6442
        self.dropout = torch.nn.Dropout2d(dropout_rate, inplace=True) if dropout_rate > 0 else torch.nn.Identity()

        if nb_filters != input_dim:
            # 1x1 conv to match the shapes (channel dimension).
            # make and build this layer separately because it directly uses input_shape
            self.shape_match_conv = torch.nn.Conv1d(input_dim, nb_filters, kernel_size=1, padding=0)
        else:
            self.shape_match_conv = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        orig_input = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.activation(x)

        x2 = self.shape_match_conv(orig_input)
        res_x = x2 + x
        res_act_x = self.activation(res_x)
        return res_act_x, x


class TCN(torch.nn.Module):
    """
    Creates a TCN layer.

    Args:
        nb_filters: The number of filters to use in the convolutional layers. Can be a list.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        padding: The padding to use in the convolutional layers, 'causal' or 'same'.
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        activation: The activation used in the residual blocks o = Activation(x + F(x)).
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        use_batch_norm: Whether to use batch normalization in the residual layers or not.
        use_layer_norm: Whether to use layer normalization in the residual layers or not.
    Returns:
        A TCN layer.
    """

    def __init__(self,
                 input_dim: int,
                 nb_filters: Union[int, Sequence[int]] = 64,
                 kernel_size: int = 3,
                 nb_stacks: int = 1,
                 dilations: List[int] = (1, 2, 4, 8, 16, 32),
                 padding: str = 'same',
                 use_skip_connections: bool = True,
                 dropout_rate: float = 0.0,
                 return_sequences: bool = False,
                 activation: Union[str, Callable] = 'relu',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False):
        super(TCN, self).__init__()

        self.return_sequences = return_sequences
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size

        if use_batch_norm + use_layer_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')

        if isinstance(nb_filters, list):
            assert len(nb_filters) == len(dilations)

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        # list to hold all the member ResidualBlocks
        residual_blocks = []
        total_num_blocks = nb_stacks * len(dilations)
        if not use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(nb_stacks):
            for i, d in enumerate(dilations):
                index = i + s * len(dilations)
                res_block_filters = nb_filters if isinstance(nb_filters, int) else nb_filters[i]
                if index == 0:
                    input_dimension = input_dim
                else:
                    input_dimension = nb_filters if isinstance(nb_filters, int) else nb_filters[i - 1]

                residual_blocks.append(TCNResidualBlock(input_dimension,
                                                        dilation_rate=d,
                                                        nb_filters=res_block_filters,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        activation=activation,
                                                        dropout_rate=dropout_rate,
                                                        use_batch_norm=use_batch_norm,
                                                        use_layer_norm=use_layer_norm))
        self.residual_blocks = torch.nn.ModuleList(residual_blocks)

    @property
    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * self.nb_stacks * sum(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, res = self.residual_blocks[0](x)
        for layer in self.residual_blocks[1:]:
            x, skip_out = layer(x)
            if self.use_skip_connections:
                res += skip_out

        if not self.use_skip_connections:
            res = x

        if not self.return_sequences:
            res = res[..., -1]

        return res
