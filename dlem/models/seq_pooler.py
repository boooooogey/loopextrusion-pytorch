import torch
from torch.nn import Module, Conv1d, BatchNorm1d, ReLU, Sequential

class InterleavedConv1d(Module):
    """InterleavedConv1d is a module that performs interleaved convolution on 1-dimensional input
    tensors. It has the right convolution which takes the odd numbered elements of the input tensor
    and the left convolution which takes the even numbered elements of the input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int):
        super(InterleavedConv1d, self).__init__()
        self.left_conv = Conv1d(in_channels=in_channels,
                                out_channels=out_channels//2,
                                kernel_size=kernel_size,
                                stride=2 * kernel_size,
                                dilation=2)
        self.right_conv = Conv1d(in_channels=in_channels,
                                 out_channels=out_channels-out_channels//2,
                                 kernel_size=kernel_size,
                                 stride=2 * kernel_size,
                                 dilation=2)
    def forward(self, x):
        left = self.left_conv(x)
        x = self.right_conv(x[...,1:])
        return torch.concat([left, x], axis=-2)

class Conv1dResidualBlock(Module):
    """Implements a residual block using convolutional layers as described in
    arxiv.org/pdf/1603.05027.
    """
    def __init__(self,
                 in_channels:int,
                 out_channels:int=64,
                 kernel_size:int=3,
                 dilation:int=2):
        super(Conv1dResidualBlock, self).__init__()
        padding = (dilation*(kernel_size - 1) + 1) // 2
        #arxiv.org/pdf/1603.05027
        self.res = Sequential(BatchNorm1d(in_channels),
                              ReLU(),
                              Conv1d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     stride=1,
                                     dilation=2,
                                     kernel_size=kernel_size,
                                     padding=padding),
                              BatchNorm1d(out_channels),
                              ReLU(),
                              Conv1d(in_channels=out_channels,
                                     out_channels=out_channels,
                                     stride=1,
                                     dilation=2,
                                     kernel_size=kernel_size,
                                     padding=padding))
    def forward(self, x):
        return self.res(x) + x

class Conv1dPoolingBlock(Module):
    """Consists of a pooling convolutional and a residual convolutional block."""
    def __init__(self,
                 in_channels:int,
                 out_channels:int=64,
                 kernel_size:int=3,
                 dilation:int=2,
                 stride:int=2):
        super(Conv1dPoolingBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.pool = Conv1d(in_channels=in_channels,
                           out_channels=out_channels,
                           stride=stride,
                           dilation=1,
                           kernel_size=kernel_size,
                           padding=padding)
        padding = (dilation*(kernel_size - 1) + 1) // 2
        #arxiv.org/pdf/1603.05027
        self.res = Conv1dResidualBlock(out_channels,
                                       out_channels,
                                       kernel_size,
                                       dilation)
    def forward(self, x):
        return self.res(self.pool(x))

class SequencePoolerInterleaved(Module):
    """The module for pooling features from sequence for HiC contact map prediction. This specific
    module uses interleaved convolution to pool the sequence features.

    Args:
        output_dim (int): The number of different features to be pooled.
        hidden_dim (int): The number of dimensions in the hidden layer.
        
    """
    def __init__(self, output_dim:int, hidden_dim:int):
        super(SequencePoolerInterleaved, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
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

    def forward(self, seq):
        return self.pooler(seq)

class SequencePoolerResidual(Module):
    """The module for pooling features from sequence for HiC contact map prediction. This specific
    module uses residual convolutional layers to pool the sequence features.

    Args:
        output_dim (int): The number of different features to be pooled.
        hidden_dim (int): The number of dimensions in the hidden layer.
        
    """
    def __init__(self, output_dim:int):
        super(SequencePoolerResidual, self).__init__()
        self.channel_numbers = [4, 32, 32, 32, 32, 64, 64, 128, output_dim]
        self.stride = [2, 2, 2, 2, 5, 5, 5, 5]
        assert len(self.channel_numbers) == len(self.stride) + 1
        def _create_conv_block(channels, kernel_size, strides):
            layers = []
            for in_channel, out_channel, stride in zip(channels[:-1], channels[1:], strides):
                layers.append(Conv1dPoolingBlock(in_channel, out_channel, kernel_size,
                                                 stride=stride))
            return Sequential(*layers)
        self.layers = _create_conv_block(self.channel_numbers, 3, self.stride)

    def forward(self, seq):
        return self.layers(seq)
    
