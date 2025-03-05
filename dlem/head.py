"""Head module blue print for the DLEM sequence models. This model produces """
from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
from torch import nn
from torch.nn import (Module,
                      Sequential,
                      ReLU,
                      Sigmoid,
                      Conv1d,
                      ConvTranspose1d,
                      ModuleList,
                      BatchNorm1d)
import torch
from dlem import util

class BaseHead(nn.Module, ABC):
    """ Base head module for the DLEM models. This module is responsible for producing a full
    contact map resolution. It may take a more complicated network, tail. 
        Args:
            size (int): one dimension of square matrix which is the contact map.
            track_dim (int): number of tracks in the input.
            seq_dim (int): number of sequence features to be mined from seqeunce or directly given
            as input.
            start_diagonal (int): starting diagonal for the contact map.
            stop_diagonal (int): stopping diagonal for the contact map.
            dlem_func (callable): DLEM function to produce the output. This can be switched to
            different modules for different physics.
            tail (nn.Module): another more complex network, probably for pooling features from a 
            sequence.
    """
    def __init__(self, size:int,
                       track_dim:int,
                       seq_dim:int,
                       start_diagonal:int,
                       stop_diagonal:int,
                       dlem_func:callable,
                       tail:nn.Module):
        super(BaseHead, self).__init__()
        self.size = size
        self.track_dim = track_dim
        self.seq_dim = seq_dim
        self.start_diagonal = start_diagonal
        self.stop_diagonal = stop_diagonal
        self.dlem = dlem_func
        self.const = nn.Parameter(torch.tensor(0.99), requires_grad = True)
        self.indexing = util.diag_index_for_mat(self.size,
                                                self.start_diagonal,
                                                self.stop_diagonal)
        self.tail = tail

    @abstractmethod
    def converter(self, tracks:torch.Tensor, seq:torch.Tensor) -> torch.Tensor:
        """Convert the input to the left and right parameters."""

    @abstractmethod
    def forward(self, diagonals:torch.Tensor,
                      tracks:torch.Tensor,
                      seq:torch.Tensor,
                      depth:int) -> torch.Tensor:
        """Forward pass of the model."""

    def dlem_output(self, diagonals:torch.Tensor,
                          left:torch.Tensor,
                          right:torch.Tensor,
                          depth:int) -> torch.Tensor:
        """After left and right parameters are produced by a model, this function calculates output
        using DLEM algorithm.

        Args:
            diagonals (torch.Tensor): diagonals of the contact maps in diagonally flatten form.
            left (torch.Tensor): left parameters of the model.
            right (torch.Tensor): right parameters of the model.
            depth (int): how many steps further to predict.

        Returns:
            torch.Tensor: Prediction for the contact map.
        """
        dev = next(self.parameters()).device
        offset = (2*self.size - 2*self.start_diagonal - depth + 1) * (depth) // 2
        out = torch.empty((diagonals.shape[0],
                           diagonals.shape[1]-offset),
                           device="cpu")
        indexing_out = util.diag_index_for_mat(self.size,
                                               self.start_diagonal+depth,
                                               self.stop_diagonal)
        for index_diag in range(self.start_diagonal, self.stop_diagonal-depth):
            diag_pred = diagonals[:, self.indexing(index_diag)]
            for _ in range(depth):
                diag_pred = torch.log(self.dlem(torch.exp(diag_pred).to(dev),
                                                left,
                                                right,
                                                self.const,
                                                self.size))
                diag_pred -=  torch.mean(diag_pred)
            out[:, indexing_out(index_diag+depth)] = diag_pred.cpu()
        return out

    def contact_map_prediction(self,
                               signal:torch.Tensor,
                               seq:torch.Tensor,
                               init_mass:torch.Tensor) -> torch.Tensor:
        """Produce the contact map

        Args:
            init_mass (torch.Tensor): initial mass for the algorithm to propagate. 

        Returns:
            torch.Tensor: predicted contact mass.
        """
        dev = next(self.parameters()).device
        curr_diag = init_mass.to(dev)
        out_len = int((self.size - (self.stop_diagonal + self.start_diagonal - 1)/2) *
                      (self.stop_diagonal-self.start_diagonal))
        out_len -= (self.size - self.start_diagonal)
        indexing_out = util.diag_index_for_mat(self.size,
                                               self.start_diagonal+1,
                                               self.stop_diagonal)
        with torch.no_grad():
            left_right = self.converter(signal.to(dev), seq.to(dev))
            left = left_right[:, 0, :]
            right = left_right[:, 1, :]
            out = torch.empty((init_mass.shape[0], out_len), device="cpu")
            for index_diag in range(self.start_diagonal, self.stop_diagonal-1):
                curr_diag = self.dlem(curr_diag,
                                      left,
                                      right,
                                      self.const,
                                      self.size)
                normed_diag = torch.log(curr_diag)
                normed_diag = normed_diag - normed_diag.mean(axis=1, keepdim=True)
                curr_diag = torch.exp(normed_diag)
                out[:, indexing_out(index_diag+1)] = normed_diag.cpu()
        return out

    def return_parameters(self, tracks:torch.Tensor,
                                seq:torch.Tensor) -> Tuple[np.ndarray,np.ndarray]:
        """Return model parameters as a tuple.

        Returns:
            Tuple[ArrayLike,ArrayLike]: parameters 
        """
        left_right = self.converter(tracks, seq)
        left = left_right[:, 0, :]
        right = left_right[:, 1, :]
        return (left.detach().cpu().numpy(),
                right.detach().cpu().numpy())

class UnetHead(BaseHead):
    """Predict contact map from Encode signals.
    """
    def __init__(self, size:int,
                       track_dim:int,
                       seq_dim:int,
                       start_diagonal:int,
                       stop_diagonal:int,
                       dlem_func:callable,
                       tail:Module,
                       network_width:Union[int,None]=None,
                       channel_per_route:int=3,
                       layer_num:int = 4):

        super(UnetHead, self).__init__(size,
                                         track_dim,
                                         seq_dim,
                                         start_diagonal,
                                         stop_diagonal,
                                         dlem_func,
                                         tail)

        def conv_unit_maker(in_dim, out_dim):
            return Sequential(Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3),
                              BatchNorm1d(out_dim),
                              ReLU())

        def trans_conv_unit_maker(in_dim, out_dim):
            return Sequential(ConvTranspose1d(in_channels=in_dim,
                                              out_channels=out_dim,
                                              kernel_size=3),
                              BatchNorm1d(out_dim),
                              ReLU())

        self.layer_num = layer_num
        self.channel_per_route = channel_per_route
        if network_width is None:
            network_width = layer_num * channel_per_route

        self.firstconv  = Sequential(Conv1d(in_channels=self.track_dim + self.seq_dim,
                                            out_channels=network_width, kernel_size=3,
                                            padding=1),
                              BatchNorm1d(network_width),
                              ReLU())
        self.convs  = [conv_unit_maker(network_width,
                                       network_width) for i in range(layer_num)]

        self.convs = ModuleList(self.convs)
        self.trans_convs = ModuleList([
            trans_conv_unit_maker(network_width,
                                  network_width) for i in range(layer_num)
            ]
        )
        self.mixer = Sequential(Conv1d(in_channels=network_width,
                                       out_channels=2,
                                       kernel_size=1),
                                Sigmoid()
                     )

    def converter(self, tracks:torch.Tensor, seq:torch.Tensor) -> torch.Tensor:
        """Converts the input epigenetic signals into DLEM parameters.

        Args:
            signal (ArrayLike): epigenetic tracks

        Returns:
            ArrayLike: parameters, p_l, p_r
        """
        tmp = self.firstconv(self.tail(torch.cat([tracks, seq], axis=-2)))
        layer_outs = [tmp]
        for n, conv in enumerate(self.convs[:-1]):
            tmp = conv(tmp)
            layer_outs.append(tmp)
        tmp = self.convs[-1](tmp)
        #maybe apply transformer later
        for n, conv in enumerate(self.trans_convs):
            tmp = conv(tmp) + layer_outs[-(n+1)]
        return self.mixer(tmp)

    def forward(self, diagonals:torch.Tensor,
                      tracks:torch.Tensor,
                      seq:torch.Tensor,
                      depth:int) -> torch.Tensor:
        """forward operation for the network.

        Args:
            curr_diag (ArrayLike): current diagonal. current state.
            diag_i (int): diagonal index for the current state.
            transform (bool, optional): Should the output converted into log space and centered.
            Defaults to True.

        Returns:
            ArrayLike: prediction for the next state.
        """
        dev = self.mixer[0].weight.device
        left_right = self.converter(tracks.to(dev), seq.to(dev))
        return self.dlem_output(diagonals, left_right[:, 0, :], left_right[:, 1, :], depth)

class ForkedHead(BaseHead):
    """Predict contact map from Encode signals.
    """
    def __init__(self, size:int,
                       track_dim:int,
                       seq_dim:int,
                       start_diagonal:int,
                       stop_diagonal:int,
                       dlem_func:callable,
                       tail:Module,
                       channel_per_route:int=3,
                       layer_num:int = 4):

        super(ForkedHead, self).__init__(size,
                                         track_dim,
                                         seq_dim,
                                         start_diagonal,
                                         stop_diagonal,
                                         dlem_func,
                                         tail)

        def conv_unit_maker(in_dim, out_dim):
            return Sequential(Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3),
                              ReLU())

        def trans_conv_unit_maker(in_dim, out_dim):
            return Sequential(ConvTranspose1d(in_channels=in_dim,
                                              out_channels=out_dim,
                                              kernel_size=3),
                              ReLU())

        self.layer_num = layer_num
        self.channel_per_route = channel_per_route
        network_width = layer_num * channel_per_route

        self.feature_batch_norm = BatchNorm1d(self.track_dim + self.seq_dim)

        self.convs  = [conv_unit_maker(self.track_dim + self.seq_dim, network_width)]
        self.convs += [conv_unit_maker(channel_per_route*i,
                                       channel_per_route*i) for i in range(layer_num-1,0,-1)]

        self.convs = ModuleList(self.convs)
        self.trans_convs = ModuleList([
            trans_conv_unit_maker(channel_per_route,
                                  channel_per_route) for i in range(layer_num)
            ]
        )
        self.mixer = Sequential(Conv1d(in_channels=network_width,
                                       out_channels=2,
                                       kernel_size=1),
                                Sigmoid()
                     )

    def converter(self, tracks:torch.Tensor, seq:torch.Tensor) -> torch.Tensor:
        """Converts the input epigenetic signals into DLEM parameters.

        Args:
            signal (ArrayLike): epigenetic tracks

        Returns:
            ArrayLike: parameters, p_l, p_r
        """
        seq = self.tail(seq)
        layer_outs = []
        tmp = self.feature_batch_norm(torch.cat([tracks, seq], axis=-2))
        for n, conv in enumerate(self.convs):
            tmp = conv(tmp)
            out_tmp = tmp[:,:self.channel_per_route]
            for trans_convs in self.trans_convs[-(n+1):]:
                out_tmp = trans_convs(out_tmp)
            layer_outs.append(out_tmp)
            tmp = tmp[:,self.channel_per_route:]
        out = torch.cat(layer_outs, axis=1)
        return self.mixer(out)

    def forward(self, diagonals:torch.Tensor,
                      tracks:torch.Tensor,
                      seq:torch.Tensor,
                      depth:int) -> torch.Tensor:
        """forward operation for the network.

        Args:
            curr_diag (ArrayLike): current diagonal. current state.
            diag_i (int): diagonal index for the current state.
            transform (bool, optional): Should the output converted into log space and centered.
            Defaults to True.

        Returns:
            ArrayLike: prediction for the next state.
        """
        dev = self.mixer[0].weight.device
        left_right = self.converter(tracks.to(dev), seq.to(dev))
        return self.dlem_output(diagonals, left_right[:, 0, :], left_right[:, 1, :], depth)

class SimpleHead(BaseHead):
    """This head doesn't learn anything and heavily relies on the tail.
    """
    def __init__(self, size:int,
                       track_dim:int,
                       seq_dim:int,
                       start_diagonal:int,
                       stop_diagonal:int,
                       dlem_func:callable,
                       tail:Module):

        super(SimpleHead, self).__init__(size,
                                         track_dim,
                                         seq_dim,
                                         start_diagonal,
                                         stop_diagonal,
                                         dlem_func,
                                         tail)
        self.norm = nn.BatchNorm1d(track_dim)

        self.activation = Sigmoid()

    def forward(self, diagonals:torch.Tensor,
                      tracks:torch.Tensor,
                      seq:torch.Tensor,
                      depth:int) -> torch.Tensor:
        """forward operation for the network.

        Args:
            curr_diag (ArrayLike): current diagonal. current state.
            diag_i (int): diagonal index for the current state.
            transform (bool, optional): Should the output converted into log space and centered.
            Defaults to True.

        Returns:
            ArrayLike: prediction for the next state.
        """
        dev = next(self.parameters()).device
        left_right = self.converter(tracks.to(dev), seq.to(dev))
        return self.dlem_output(diagonals, left_right[:, 0, :], left_right[:, 1, :], depth)

    def converter(self, tracks:torch.Tensor, seq:torch.Tensor) -> torch.Tensor:
        """Convert the input to the left and right parameters."""
        tracks = self.norm(tracks)
        left_right = self.tail(torch.concatenate([tracks, seq], axis=1))
        return self.activation(left_right)

class ForkedBasePairTrackHeadBinary(BaseHead):
    """Predict contact map from Encode signals.
    """
    def __init__(self, size:int,
                       track_dim:int,
                       seq_dim:int,
                       start_diagonal:int,
                       stop_diagonal:int,
                       dlem_func:callable,
                       tail:Module,
                       channel_per_route:int=3,
                       layer_num:int = 6):

        super(ForkedBasePairTrackHeadBinary, self).__init__(size,
                                         track_dim,
                                         seq_dim,
                                         start_diagonal,
                                         stop_diagonal,
                                         dlem_func,
                                         tail)

        def conv_unit_maker(in_dim, out_dim):
            return Sequential(Conv1d(in_channels=in_dim,
                                     out_channels=out_dim, kernel_size=3, dilation=2, padding=2),
                              ReLU(),
                              Conv1d(in_channels=out_dim,
                                     out_channels=out_dim,
                                     groups=out_dim, kernel_size=2, stride=2),
                              ReLU())

        def trans_conv_unit_maker(in_dim, out_dim):
            return Sequential(ConvTranspose1d(in_channels=in_dim,
                                              out_channels=out_dim, kernel_size=2, stride=2),
                              ReLU(),
                              ConvTranspose1d(in_channels=out_dim,
                                              out_channels=out_dim,
                                              groups=out_dim,
                                              kernel_size=3, dilation=2, padding=2),
                              ReLU())

        self.layer_num = layer_num
        self.channel_per_route = channel_per_route
        network_width = layer_num * channel_per_route

        self.feature_batch_norm = BatchNorm1d(self.track_dim + self.seq_dim)

        self.convs  = [conv_unit_maker(self.track_dim + self.seq_dim, network_width)]
        self.convs += [conv_unit_maker(channel_per_route*i,
                                       channel_per_route*i) for i in range(layer_num-1,0,-1)]

        self.convs = ModuleList(self.convs)
        self.trans_convs = ModuleList([
            trans_conv_unit_maker(channel_per_route,
                                  channel_per_route) for i in range(layer_num)
            ]
        )
        self.mixer = Sequential(Conv1d(in_channels=network_width,
                                       out_channels=2,
                                       kernel_size=1),
                                Sigmoid()
                     )

        self.raw_track_norm = nn.BatchNorm1d(track_dim)

    def converter(self, tracks:torch.Tensor, seq:torch.Tensor) -> torch.Tensor:
        """Converts the input epigenetic signals into DLEM parameters.

        Args:
            signal (ArrayLike): epigenetic tracks

        Returns:
            ArrayLike: parameters, p_l, p_r
        """
        layer_outs = []
        #difference with ForkedHead is that we are using the tail for the tracks too.
        tmp = self.tail(torch.cat([self.raw_track_norm(tracks), seq], axis=-2))
        tmp = self.feature_batch_norm(tmp)
        for n, conv in enumerate(self.convs):
            tmp = conv(tmp)
            out_tmp = tmp[:,:self.channel_per_route]
            for trans_convs in self.trans_convs[-(n+1):]:
                out_tmp = trans_convs(out_tmp)
            layer_outs.append(out_tmp)
            tmp = tmp[:,self.channel_per_route:]
        out = torch.cat(layer_outs, axis=1)
        return self.mixer(out)

    def forward(self, diagonals:torch.Tensor,
                      tracks:torch.Tensor,
                      seq:torch.Tensor,
                      depth:int) -> torch.Tensor:
        """forward operation for the network.

        Args:
            curr_diag (ArrayLike): current diagonal. current state.
            diag_i (int): diagonal index for the current state.
            transform (bool, optional): Should the output converted into log space and centered.
            Defaults to True.

        Returns:
            ArrayLike: prediction for the next state.
        """
        dev = self.mixer[0].weight.device
        left_right = self.converter(tracks.to(dev), seq.to(dev))
        return self.dlem_output(diagonals, left_right[:, 0, :], left_right[:, 1, :], depth)

class ForkedBasePairTrackHeadBinaryUnet(BaseHead):
    """Predict contact map from Encode signals.
    """
    def __init__(self, size:int,
                       track_dim:int,
                       seq_dim:int,
                       start_diagonal:int,
                       stop_diagonal:int,
                       dlem_func:callable,
                       tail:Module,
                       channel_per_route:int=3,
                       layer_num:int = 6):

        super(ForkedBasePairTrackHeadBinaryUnet, self).__init__(size,
                                         track_dim,
                                         seq_dim,
                                         start_diagonal,
                                         stop_diagonal,
                                         dlem_func,
                                         tail)

        def conv_unit_maker(in_dim, out_dim):
            return Sequential(Conv1d(in_channels=in_dim,
                                     out_channels=out_dim, kernel_size=3, dilation=2, padding=2),
                              ReLU(),
                              Conv1d(in_channels=out_dim,
                                     out_channels=out_dim,
                                     groups=out_dim, kernel_size=2, stride=2),
                              ReLU())

        def trans_conv_unit_maker(in_dim, out_dim):
            return Sequential(ConvTranspose1d(in_channels=in_dim,
                                              out_channels=out_dim, kernel_size=2, stride=2),
                              ReLU(),
                              ConvTranspose1d(in_channels=out_dim,
                                              out_channels=out_dim,
                                              groups=out_dim,
                                              kernel_size=3, dilation=2, padding=2),
                              ReLU())

        self.layer_num = layer_num
        self.channel_per_route = channel_per_route
        network_width = layer_num * channel_per_route

        self.feature_batch_norm = BatchNorm1d(self.track_dim + self.seq_dim)

        self.conv_first = Sequential(Conv1d(in_channels=self.track_dim + self.seq_dim,
                                            out_channels=network_width,
                                            kernel_size=3, dilation=2, padding=2),
                                    ReLU())

        self.convs  = [conv_unit_maker(network_width, network_width)]
        self.convs += [conv_unit_maker(network_width,
                                       network_width) for i in range(layer_num-1,0,-1)]

        self.convs = ModuleList(self.convs)
        self.trans_convs = ModuleList([
            trans_conv_unit_maker(network_width,
                                  network_width) for i in range(layer_num)
            ]
        )
        self.mixer = Sequential(Conv1d(in_channels=network_width,
                                       out_channels=2,
                                       kernel_size=1),
                                Sigmoid()
                     )

        self.raw_track_norm = nn.BatchNorm1d(track_dim)

    def converter(self, tracks:torch.Tensor, seq:torch.Tensor) -> torch.Tensor:
        """Converts the input epigenetic signals into DLEM parameters.

        Args:
            signal (ArrayLike): epigenetic tracks

        Returns:
            ArrayLike: parameters, p_l, p_r
        """
        layer_outs = []
        #difference with ForkedHead is that we are using the tail for the tracks too.
        tmp = self.tail(torch.cat([self.raw_track_norm(tracks), seq], axis=-2))
        tmp = self.feature_batch_norm(tmp)
        tmp = self.conv_first(tmp)
        layer_outs = [tmp]
        for n, conv in enumerate(self.convs[:-1]):
            tmp = conv(tmp)
            layer_outs.append(tmp)
        tmp = self.convs[-1](tmp)
        for n, trans_convs in enumerate(self.trans_convs):
            tmp = trans_convs(tmp) + layer_outs[-(n+1)]
        return self.mixer(tmp)

    def forward(self, diagonals:torch.Tensor,
                      tracks:torch.Tensor,
                      seq:torch.Tensor,
                      depth:int) -> torch.Tensor:
        """forward operation for the network.

        Args:
            curr_diag (ArrayLike): current diagonal. current state.
            diag_i (int): diagonal index for the current state.
            transform (bool, optional): Should the output converted into log space and centered.
            Defaults to True.

        Returns:
            ArrayLike: prediction for the next state.
        """
        dev = self.mixer[0].weight.device
        left_right = self.converter(tracks.to(dev), seq.to(dev))
        return self.dlem_output(diagonals, left_right[:, 0, :], left_right[:, 1, :], depth)

class ForkedBasePairTrackHead(ForkedHead):
    """Predict contact map from Encode signals.
    """
    def __init__(self, size:int,
                       track_dim:int,
                       seq_dim:int,
                       start_diagonal:int,
                       stop_diagonal:int,
                       dlem_func:callable,
                       tail:Module,
                       channel_per_route:int=3,
                       layer_num:int = 4):

        output_dim = tail.channel_numbers[-1]

        super(ForkedBasePairTrackHead, self).__init__(size,
                                         1,
                                         output_dim-1,
                                         start_diagonal,
                                         stop_diagonal,
                                         dlem_func,
                                         tail,
                                         channel_per_route,
                                         layer_num)
        
        self.raw_track_norm = nn.BatchNorm1d(track_dim)

    def converter(self, tracks:torch.Tensor, seq:torch.Tensor) -> torch.Tensor:
        """Converts the input epigenetic signals into DLEM parameters.

        Args:
            signal (ArrayLike): epigenetic tracks

        Returns:
            ArrayLike: parameters, p_l, p_r
        """
        layer_outs = []
        #difference with ForkedHead is that we are using the tail for the tracks too.
        tmp = self.tail(torch.cat([self.raw_track_norm(tracks), seq], axis=-2))
        tmp = self.feature_batch_norm(tmp)
        for n, conv in enumerate(self.convs):
            tmp = conv(tmp)
            out_tmp = tmp[:,:self.channel_per_route]
            for trans_convs in self.trans_convs[-(n+1):]:
                out_tmp = trans_convs(out_tmp)
            layer_outs.append(out_tmp)
            tmp = tmp[:,self.channel_per_route:]
        out = torch.cat(layer_outs, axis=1)
        return self.mixer(out)

class ForkedBasePairTrackHeadSimple(ForkedHead):
    """Predict contact map from Encode signals.
    """
    def __init__(self, size:int,
                       track_dim:int,
                       seq_dim:int,
                       start_diagonal:int,
                       stop_diagonal:int,
                       dlem_func:callable,
                       tail:Module,
                       channel_per_route:int=3,
                       layer_num:int = 4):

        output_dim = tail.channel_numbers[-1]

        super(ForkedBasePairTrackHeadSimple, self).__init__(size,
                                         1,
                                         output_dim-1,
                                         start_diagonal,
                                         stop_diagonal,
                                         dlem_func,
                                         tail,
                                         channel_per_route,
                                         layer_num)
        
        self.raw_track_norm = nn.BatchNorm1d(track_dim + 2)

    def converter(self, tracks:torch.Tensor, seq:torch.Tensor) -> torch.Tensor:
        """Converts the input epigenetic signals into DLEM parameters.

        Args:
            signal (ArrayLike): epigenetic tracks

        Returns:
            ArrayLike: parameters, p_l, p_r
        """
        layer_outs = []
        #difference with ForkedHead is that we are using the tail for the tracks too.
        tmp = self.tail(self.raw_track_norm(torch.cat([tracks, seq], axis=-2)))
        tmp = self.feature_batch_norm(tmp)
        for n, conv in enumerate(self.convs):
            tmp = conv(tmp)
            out_tmp = tmp[:,:self.channel_per_route]
            for trans_convs in self.trans_convs[-(n+1):]:
                out_tmp = trans_convs(out_tmp)
            layer_outs.append(out_tmp)
            tmp = tmp[:,self.channel_per_route:]
        out = torch.cat(layer_outs, axis=1)
        return self.mixer(out)


class MinHead(BaseHead):
    """Predict contact map from Encode signals.
    """
    def __init__(self, size:int,
                       track_dim:int,
                       seq_dim:int,
                       start_diagonal:int,
                       stop_diagonal:int,
                       dlem_func:callable,
                       tail:Module,
                       channel_per_route:int=3,
                       layer_num:int = 4):

        sample = 10

        super(MinHead, self).__init__(size,
                                      track_dim,
                                      seq_dim,
                                      start_diagonal,
                                      stop_diagonal,
                                      dlem_func,
                                      tail)

        self.pool = nn.MaxPool1d(sample)

    def converter(self, tracks, seq):
        left_right = self.tail(seq, tracks)
        return -self.pool(-left_right)

    def forward(self, diagonals:torch.Tensor,
                      tracks:torch.Tensor,
                      seq:torch.Tensor,
                      depth:int) -> torch.Tensor:
        dev = next(self.tail.parameters()).device
        left_right = self.converter(tracks.to(dev), seq.to(dev))
        return self.dlem_output(diagonals, left_right[:, 0, :], left_right[:, 1, :], depth)

    def return_res(self,
                   scale:float,
                   signal:torch.Tensor,
                   seq:torch.Tensor,
                   init_mass:torch.Tensor) -> torch.Tensor:
        """Produce the contact map

        Args:
            init_mass (torch.Tensor): initial mass for the algorithm to propagate. 

        Returns:
            torch.Tensor: predicted contact mass.
        """
        dev = next(self.parameters()).device
        curr_diag = init_mass.to(dev)
        stop_diagonal = int(self.stop_diagonal * scale)
        start_diagonal = int(self.start_diagonal * scale)
        size = int(self.size * scale)
        out_len = int((size - (stop_diagonal + start_diagonal - 1)/2) *
                      (stop_diagonal-start_diagonal))
        out_len -= (size - start_diagonal)
        indexing_out = util.diag_index_for_mat(size,
                                               start_diagonal+1,
                                               stop_diagonal)
        if self.pool.kernel_size // scale == 1:
            max_pool = nn.Identity()
        else:
            max_pool = nn.MaxPool1d(self.pool.kernel_size//scale)

        def converter(tracks, seq):
            left_right = self.tail(seq, tracks)
            return -max_pool(-left_right)

        with torch.no_grad():
            left_right = converter(signal.to(dev), seq.to(dev))
            left = left_right[:, 0, :]
            right = left_right[:, 1, :]
            out = torch.empty((init_mass.shape[0], out_len), device="cpu")
            for index_diag in range(start_diagonal, stop_diagonal-1):
                curr_diag = self.dlem(curr_diag,
                                      left,
                                      right,
                                      self.const,
                                      size)
                normed_diag = torch.log(curr_diag)
                normed_diag = normed_diag - normed_diag.mean(axis=1, keepdim=True)
                curr_diag = torch.exp(normed_diag)
                out[:, indexing_out(index_diag+1)] = normed_diag.cpu()
        return out, left_right
