"""The module for pooling features from sequence for HiC contact map prediction."""
from typing import List
from torch.nn import Module, Conv1d, Sequential, GELU
import torch
from dlem.trunks.attentionpooling import AttentionPooling1D
from dlem.trunks.conv1d_layers import Conv1dResidualBlock, Conv1dPoolingBlock, InterleavedConv1d

class SequencePoolerInterleaved(Module):
    """The module for pooling features from sequence for HiC contact map prediction. This specific
    module uses interleaved convolution to pool the sequence features.

    Args:
        output_dim (int): The number of different features to be pooled.
        hidden_dim (int): The number of dimensions in the hidden layer.
    """
    def __init__(self, channel_numbers:List[int], stride:List[int]):
        super(SequencePoolerInterleaved, self).__init__()
        self.output_dim = channel_numbers[-1]#output_dim
        self.hidden_dim = channel_numbers[0]#hidden_dim
        self.strid = stride #not used
        layers = [InterleavedConv1d(in_channels=4,
                                    out_channels=self.hidden_dim,
                                    kernel_size=5),
                  Conv1dResidualBlock(in_channels=self.hidden_dim,
                                      out_channels=self.hidden_dim)]

        layers += [InterleavedConv1d(in_channels=self.hidden_dim,
                                     out_channels=self.hidden_dim,
                                     kernel_size=5),
                   Conv1dResidualBlock(in_channels=self.hidden_dim,
                                       out_channels=self.hidden_dim)] * 2

        layers += [InterleavedConv1d(in_channels=self.hidden_dim,
                                     out_channels=self.output_dim,
                                     kernel_size=5),
                   Conv1dResidualBlock(in_channels=self.output_dim,
                                       out_channels=self.output_dim)]

        self.pooler = Sequential(*layers)

    def forward(self, seq:torch.Tensor) -> torch.Tensor:
        """forward function.

        Args:
            seq (torch.Tensor): input sequence.

        Returns:
            torch.Tensor: sequence features.
        """
        return self.pooler(seq)

class SequencePoolerResidual(Module):
    """The module for pooling features from sequence for HiC contact map prediction. This specific
    module uses residual convolutional layers to pool the sequence features.

    Args:
        output_dim (int): The number of different features to be pooled.
        
    """
    def __init__(self, channel_numbers, stride):
        super(SequencePoolerResidual, self).__init__()
        self.channel_numbers = channel_numbers #[4, 32, 32, 32, 32, 64, 64, 128, output_dim]
        self.stride = stride #[2, 2, 2, 2, 5, 5, 5, 5]
        assert len(self.channel_numbers) == len(self.stride) + 1
        def _create_conv_block(channels, kernel_size, strides):
            layers = []
            for in_channel, out_channel, stride in zip(channels[:-1], channels[1:], strides):
                layers.append(Conv1dPoolingBlock(in_channel, out_channel, kernel_size,
                                                 stride=stride))
            return Sequential(*layers)
        self.layers = _create_conv_block(self.channel_numbers, 3, self.stride)

    def forward(self, seq):
        """forward function.

        Args:
            seq (torch.Tensor): input sequence.

        Returns:
            torch.Tensor: sequence features.
        """
        return self.layers(seq)

class SequencePoolerAttention(Module):
    """The module for pooling features from sequence for HiC contact map prediction. This specific
    module uses residual convolutional layers to pool the sequence features.

    Args:
        output_dim (int): The number of different features to be pooled.
    """
    def __init__(self, channel_numbers, stride):
        super(SequencePoolerAttention, self).__init__()
        self.channel_numbers = channel_numbers #[4, 16, 16, 16, output_dim]
        self.stride = stride #[10, 10, 10, 10]
        assert len(self.channel_numbers) == len(self.stride) + 1
        def _create_conv_block(channels, kernel_size, strides):
            layers = []
            for in_channel, out_channel, stride in zip(channels[:-1], channels[1:], strides):
                layers.append(Conv1d(in_channels=in_channel,
                                     out_channels=out_channel,
                                     kernel_size=1))
                layers.append(Conv1dResidualBlock(out_channel,
                                                  out_channels=out_channel,
                                                  kernel_size=kernel_size,
                                                  dilation=2,
                                                  activation_func=GELU))
                layers.append(AttentionPooling1D(stride, out_channel, mode="full"))
            return Sequential(*layers)
        self.layers = _create_conv_block(self.channel_numbers, 3, self.stride)

    def forward(self, seq):
        """forward function.

        Args:
            seq (torch.Tensor): input sequence.

        Returns:
            torch.Tensor: sequence features.
        """
        return self.layers(seq)
    