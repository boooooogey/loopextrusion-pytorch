"""The module for pooling features from sequence for HiC contact map prediction."""
from abc import ABC, abstractmethod
from typing import List
import importlib.resources as pkg_resources
from torch.nn import Module, Conv1d, ConvTranspose1d, Sequential, GELU, ModuleList
import torch
import numpy as np
import einops
from dlem.attentionpooling import AttentionPooling1D
from dlem.conv1d_layers import Conv1dResidualBlock, Conv1dPoolingBlock, InterleavedConv1d

class SequencePooler(Module, ABC):
    """Blueprint for pooling features from sequence for HiC contact map prediction.

    Args:
        channel_numbers (List[int]): the number of channels in the convolutional layers.
        stride (List[int]): specifies how to shrink features down to corresponding bin size.
    """
    def __init__(self, channel_numbers:List[int], stride:List[int]):
        super(SequencePooler, self).__init__()
        self.channel_numbers = channel_numbers
        self.stride = stride

    @abstractmethod
    def forward(self, seq:torch.Tensor) -> torch.Tensor:
        """Forward pass function

        Args:
            seq (torch.Tensor): sequence tensor. 

        Returns:
            torch.Tensor: features
        """

class SequencePoolerInterleaved(SequencePooler):
    """The module for pooling features from sequence for HiC contact map prediction. This specific
    module uses interleaved convolution to pool the sequence features.

    Args:
        output_dim (int): The number of different features to be pooled.
        hidden_dim (int): The number of dimensions in the hidden layer.
    """
    def __init__(self, channel_numbers:List[int], stride:List[int]):
        super(SequencePoolerInterleaved, self).__init__(channel_numbers, stride)
        self.output_dim = self.channel_numbers[-1]#output_dim
        self.hidden_dim = self.channel_numbers[0]#hidden_dim
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

class SequencePoolerResidual(SequencePooler):
    """The module for pooling features from sequence for HiC contact map prediction. This specific
    module uses residual convolutional layers to pool the sequence features.

    Args:
       channel_numbers (List[int]): the number of channels in the convolutional layers. 
       Example: [4, 32, 32, 32, 32, 64, 64, 128, output_dim].
       stride (List[int]): specifies how to shrink features down to corresponding bin size.
       Example: [2, 2, 2, 2, 5, 5, 5, 5].
    """
    def __init__(self, channel_numbers:List[int], stride:List[int]):
        super(SequencePoolerResidual, self).__init__(channel_numbers, stride)
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

class SequencePoolerAttention(SequencePooler):
    """The module for pooling features from sequence for HiC contact map prediction. This specific
    module uses residual convolutional layers to pool the sequence features.

    Args:
        channel_numbers (List[int]): the number of channels in the convolutional layers.
        Example: [4, 16, 16, 16, output_dim]
        stride (List[int]): specifies how to shrink features down to corresponding bin size.
        Example: [10, 10, 10, 10]
    """
    def __init__(self, channel_numbers:List[int], stride:List[int]):
        super(SequencePoolerAttention, self).__init__(channel_numbers, stride)
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
    
class SequencePoolerCTCF(SequencePooler):
    """Uses CTCF motif to pool the sequence features.

    Args:
        SequencePooler (_type_): _description_
    """
    def __init__(self, channel_numbers:List[int], stride:List[int]):
        super(SequencePoolerCTCF, self).__init__(channel_numbers, stride)
        with pkg_resources.path("dlem", "ctcf_kernels_jaspar.npy") as path:
            kernels = np.load(path)
        self.ctcf_conv = Conv1d(in_channels = kernels.shape[1],
                                out_channels = kernels.shape[0],
                                kernel_size = kernels.shape[2],
                                padding=kernels.shape[2]//2)
        self.ctcf_conv.weight = torch.nn.Parameter(torch.from_numpy(kernels).float())
        self.ctcf_conv.weight.requires_grad = False
        self.ctcf_scale = torch.nn.Parameter(torch.ones(kernels.shape[0]), requires_grad=True)
        self.ctcf_bias = torch.nn.Parameter(torch.zeros(1, kernels.shape[0], 1), requires_grad=True)

        #num_channel = 4 + kernels.shape[0]

        self.conv_chrom_access = Sequential(Conv1d(
            in_channels=1,
            out_channels=self.channel_numbers[0] - kernels.shape[0],
            kernel_size=kernels.shape[2],
            padding=kernels.shape[2]//2),
                                            GELU())

        self.attention_pooling = AttentionPooling1D(1000, self.channel_numbers[0], mode="full")

        layers = []
        for cn, st in zip(self.channel_numbers, self.stride):
            layers.append(Sequential(Conv1d(in_channels=cn,
                                      out_channels=cn,
                                      kernel_size=st),
                                     GELU()))
        self.conv_forward = ModuleList(layers)
        layers = []
        for cn, st in zip(self.channel_numbers, self.stride):
            layers.append(Sequential(ConvTranspose1d(in_channels=cn,
                                      out_channels=cn,
                                      kernel_size=st),
                                     GELU()))

        self.conv_backward = ModuleList(layers)

        self.mix = Conv1d(in_channels=self.channel_numbers[-1],
                          out_channels=2,
                          kernel_size=1)

    def forward(self, seq, chrom_access):

        x = einops.einsum(self.ctcf_conv(seq),
                          self.ctcf_scale, "b c w, c -> b c w") + self.ctcf_bias

        x = torch.concat([self.conv_chrom_access(chrom_access), x], axis=1)

        x = self.attention_pooling(x)

        forward_pass = [x]
        for layer in self.conv_forward[:-1]:
            x = layer(x)
            forward_pass.append(x)
        x = self.conv_forward[-1](x)
        for fp, layer in zip(forward_pass[::-1], self.conv_backward):
            x = layer(x) + fp
        return torch.sigmoid(self.mix(x))
