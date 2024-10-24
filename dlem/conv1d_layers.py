"""One dimensional convolutional layers used for the DLEM sequence feature pooling. 
"""
from torch.nn import Module, Conv1d, BatchNorm1d, ReLU, Sequential
import torch

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
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): input. 

        Returns:
            torch.Tensor: output.
        """
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
                 dilation:int=2,
                 activation_func:callable=ReLU):
        super(Conv1dResidualBlock, self).__init__()
        padding = (dilation*(kernel_size - 1) + 1) // 2
        #arxiv.org/pdf/1603.05027
        self.res = Sequential(BatchNorm1d(in_channels),
                              activation_func(),
                              Conv1d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     stride=1,
                                     dilation=2,
                                     kernel_size=kernel_size,
                                     padding=padding),
                              BatchNorm1d(out_channels),
                              activation_func(),
                              Conv1d(in_channels=out_channels,
                                     out_channels=out_channels,
                                     stride=1,
                                     dilation=2,
                                     kernel_size=kernel_size,
                                     padding=padding))
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Residual block forward function.

        Args:
            x (torch.Tensor): input.

        Returns:
            torch.Tensor: output that has the same shape as the input.
        """
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
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: outpu 
        """
        return self.res(self.pool(x))
