"""
Implementations of basic recurrent neural networks.
"""
from typing import List, Union, Optional, Tuple, Sequence

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, layer_type: str, model: str, input_dimension: int, hidden_dimensions: Union[Sequence[int], int],
                 n_recurrent_layers: Optional[int] = None, recurrent_activation: str = 'tanh',
                 recurrent_bias: bool = True, bidirectional: bool = False, projection_size: int = 0,
                 dilation: Optional[Sequence[int]] = None, dropout: float = 0.0):
        """
        General framework of a recurrent neural network.

        :param layer_type: The type of recurrent layer (RNN, LSTM, GRU).
        :type layer_type: str
        :param model: Type of model (s2s, s2as, s2fh, s2mh)
        :type model: str
        :param input_dimension: Number of dimensions of the feature space of the input data.
        :type input_dimension: int
        :param hidden_dimensions: Number of dimensions of each hidden state for all recurrent layers.
        :type hidden_dimensions: Union[List[int], int]
        :param n_recurrent_layers: Number of recurrent layers, if hidden_dimensions is the same for all layers (int).
        :type n_recurrent_layers: Optional[int]
        :param recurrent_activation: Activation function to use in each recurrent layer (has to be defined in torch.nn).
        :type recurrent_activation: str
        :param recurrent_bias: Whether to use bias in the computations of the recurrent layers.
        :type recurrent_bias: bool
        :param bidirectional: Use bidirectional recurrent layers.
        :type bidirectional: bool
        :param projection_size: Parameter relevant for LSTM layers only. See pytorch docs for details.
        :type projection_size: int
        """

        super(RNN, self).__init__()

        if isinstance(hidden_dimensions, int):
            if n_recurrent_layers is None:
                hidden_dimensions = [hidden_dimensions]
            else:
                hidden_dimensions = [hidden_dimensions for _ in range(n_recurrent_layers)]

        if dilation is None:
            dilation = [0 for _ in hidden_dimensions]
        if len(dilation) != len(hidden_dimensions):
            raise ValueError('Dilation must match the number of hidden layers!')
        self.dilation = dilation

        if layer_type.lower() == 'rnn':
            layer = nn.RNN
        elif layer_type.lower() == 'lstm':
            layer = nn.LSTM
        elif layer_type.lower() == 'gru':
            layer = nn.GRU
        elif callable(layer_type):
            layer = layer_type
        else:
            raise NotImplementedError(layer_type)

        layer_kwargs = {'num_layers': 1, 'bias': recurrent_bias,
                        'batch_first': False,
                        'bidirectional': bidirectional}

        if layer_type.lower() == 'lstm':
            layer_kwargs['proj_size'] = projection_size
        elif layer_type.lower() == 'rnn':
            layer_kwargs['nonlinearity'] = recurrent_activation

        self.dropout = torch.nn.Identity() if dropout <= 0.0 else torch.nn.Dropout(dropout)

        dims = [input_dimension] + list(hidden_dimensions)
        self.recurrent_layers = nn.ModuleList([layer(input_size=input_dim, hidden_size=hidden_dim, **layer_kwargs)
                                               for input_dim, hidden_dim in zip(dims[:-1], dims[1:])])

        self.model = model

    def _base_rnn_forward(self,
                          inputs: Union[torch.Tensor, nn.utils.rnn.PackedSequence],
                          hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                          return_all_sequences: bool = False) \
            -> Tuple[Union[List[torch.Tensor], torch.Tensor, nn.utils.rnn.PackedSequence], ...]:
        """
        Apply the base RNN to the sequence and return the hidden states from every step in the sequence.

        :param inputs: Tensor or sequence of shape (T, B, ...)
        :param hidden_states: Hidden states of shape (num_layers, B, ...). For LSTM this must be a tuple of two tensors.
        :param return_all_sequences: Whether the method should return outputs of intermediate layers, too
        :return: If return_all_sequences=False this is a tensor or PackedSequence of shape (T, B, ...),
            otherwise a list of such tensors with one element for each layer is returned.
        """

        # LSTM needs two tensors for hidden state
        if isinstance(hidden_states, tuple):
            hidden_states = zip(*hidden_states)
        elif hidden_states is None:
            hidden_states = (None for _ in self.recurrent_layers)

        layer_output = inputs
        layer_hidden = None
        outputs = []
        hiddens = []

        for i, (layer, dilation, hidden) in enumerate(zip(self.recurrent_layers, self.dilation, hidden_states)):
            if isinstance(hidden, tuple):
                hidden = tuple(torch.unsqueeze(h, dim=0) for h in hidden)
            elif isinstance(hidden, torch.Tensor):
                hidden = torch.unsqueeze(hidden, dim=0)

            if dilation > 1:
                layer_output, layer_hidden = self.apply_dilated_layer(layer, layer_output, dilation, hidden)
            else:
                layer_output, layer_hidden = layer(layer_output, hidden)

            if i < len(self.recurrent_layers) - 1:
                # Apply dropout, but not after the last layer
                layer_output = self.dropout(layer_output)

            if return_all_sequences:
                outputs.append(layer_output)
                hiddens.append(layer_hidden)

        if return_all_sequences:
            return outputs, hiddens

        return layer_output, layer_hidden

    def apply_dilated_layer(self, layer: torch.nn.Module,
                            inputs: Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence], dilation: int,
                            hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence],
                  Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        # TODO: handle packed sequence
        inputs, padding, hidden = self._dilate_inputs(inputs, dilation, hidden)

        outputs, hidden = layer(inputs, hidden)

        outputs = self._restore_outputs(outputs, dilation, padding)

        return outputs, hidden

    @staticmethod
    def _restore_outputs(dilated_outputs: torch.Tensor, dilation: int, input_padding: int) -> torch.Tensor:
        """
        Restores output of a RNN applied to a sequence prepared with _dilate_inputs to its original form.
        That means the subsampled sequences are interleaved again and padding is removed from the end of the sequence.

        :param dilated_outputs: Tensor of shape ((T+padding)//dilation, B*dilation, ...)
        :param dilation: Dilation (subsampling) factor
        :param input_padding: Padding applied to the input sequence before applying the RNN
        :return: Tensor of size (T, B, ...)
        """
        in_shape = dilated_outputs.shape
        batch_size = in_shape[1] // dilation

        blocks = [dilated_outputs[:, i * batch_size: (i + 1) * batch_size] for i in range(dilation)]

        interleaved = torch.stack(blocks).transpose(1, 0).contiguous()
        out_shape = [in_shape[0] * dilation, batch_size] + list(interleaved.shape[3:])
        interleaved = interleaved.view(out_shape)
        if input_padding > 0:
            interleaved = interleaved[:-input_padding]

        return interleaved

    @staticmethod
    def _dilate_inputs(inputs: torch.Tensor, dilation: int, hidden: torch.Tensor = None) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Subsamples the input sequence with a factor of dilation. Each of the first dilation elements is used as
        a starting point for such a subsampled sequence and they are stacked along the batch axis. Note that the
        sequence will be padded with zeros at the end so that its length is a multiple of dilation.

        :param inputs: Tensor of shape (T, B, ...), where T is the sequence length and N the batch size
        :param dilation: Dilation (subsampling) factor
        :param hidden: Tensor of shape (B, hidden_size) containing initial hidden state or a tuple of two such tensors
            in case of the LSTM
        :return: Tensor of shape ((T+padding)//dilation, N*dilation, ...) and the number of time-steps that were
            applied as padding and the new hidden state
        """
        seq_len = inputs.shape[0]
        padding = (seq_len + dilation - 1) // dilation * dilation - seq_len

        if padding > 0:
            pad_shape = [padding] + list(inputs.shape[1:])
            zeros_ = torch.zeros(pad_shape, device=inputs.device, dtype=inputs.dtype)
            inputs = torch.cat((inputs, zeros_))

        dilated_inputs = torch.cat([inputs[j::dilation] for j in range(dilation)], 1)

        if hidden is not None:
            hidden = hidden.repeat(dilation, 1)

        return dilated_inputs, padding, hidden

    def forward(self, inputs: Union[torch.Tensor, nn.utils.rnn.PackedSequence],
                hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                return_hidden: bool = False) \
            -> Union[Tuple[torch.Tensor, ...], torch.Tensor, nn.utils.rnn.PackedSequence]:
        """
        Apply the base RNN to the sequence and return the hidden states from every step in the sequence.

        :param inputs: Tensor or sequence of shape (T, B, ...)
        :param hidden_states: Hidden states of shape (num_layers, B, ...). For LSTM this must be a tuple of two tensors.
        :param return_hidden: Whether the method should return hidden states of the RNN too
        :return: If return_hidden=False this is a tensor or PackedSequence of shape (T, B, ...),
            otherwise a Tuple (output, hidden)
        """
        return_all = False
        if self.model == 's2as':
            return_all = True

        output, hidden = self._base_rnn_forward(inputs, hidden_states, return_all_sequences=return_all)

        if self.model in ['s2s', 's2as']:
            pass
        elif self.model == 's2fh':
            output = output[-1]
        elif self.model == 's2mh':
            output = torch.mean(output, dim=0)
        else:
            raise NotImplementedError(f'Model {self.model} is not supported!')

        if return_hidden:
            return output, hidden
        else:
            return output
