"""Implementation of DLEM with pytorch
"""
from typing import Tuple
from numpy.typing import ArrayLike
import torch
from torch.nn import (Module,
                      Parameter,
                      Sequential,
                      ReLU,
                      Sigmoid,
                      Conv1d,
                      ConvTranspose1d,
                      ModuleList,
                      BatchNorm1d)
from .. import util
from .seq_pooler import SequencePoolerResidual as SequencePooler

def _dlem(curr_diag:ArrayLike,
          left_right:ArrayLike,
          const:ArrayLike,
          index_diag:int,
          n:int,
          eps=1e-6) -> ArrayLike:
    diag_len = curr_diag.shape[1]

    left = left_right[:, 0, :]
    right = left_right[:, 1, :]

    index_in_left = range(index_diag, n-1)
    index_in_right = range(1, n-index_diag)
    index_out_left = range(index_diag+1, n)
    index_out_right = range(0, n-index_diag-1)

    index_curr_diag_left = range(0, diag_len-1)
    index_curr_diag_right = range(1, diag_len)

    mass_in = curr_diag[:, index_curr_diag_right] * right[:, index_in_right]
    mass_in += curr_diag[:, index_curr_diag_left] * left[:, index_in_left]

    mass_out = right[:, index_out_right] + left[:, index_out_left] + eps

    next_diag_pred = const * mass_in / mass_out

    return next_diag_pred

class DLEM(Module):
    """Predict contact map from Encode signals.
    """
    def __init__(self, n:int,
                       epi_dim:int,
                       start_diag:int,
                       stop_diag:int,
                       seq_fea_dim:int,
                       channel_per_route:int=3,
                       layer_num:int = 4):
        """_summary_

        Args:
            left_init (ArrayLike): Initiation for the left parameters.
            right_init (ArrayLike): Initiation for the right parameters.
            free_unload (bool, optional): _description_. Defaults to False.
            type (int, optional): _description_. Defaults to 1.
        """
        super(DLEM, self).__init__()

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
        self.seq_fea_dim = seq_fea_dim
        self.epi_dim = epi_dim
        self.seq_pooler = SequencePooler(seq_fea_dim)

        self.feature_batch_norm = BatchNorm1d(epi_dim + seq_fea_dim)

        self.convs  = [conv_unit_maker(epi_dim + seq_fea_dim, network_width)]
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

        self.n = n
        self.const = Parameter(torch.tensor(0.99), requires_grad = True)
        self.start_diag = start_diag
        self.stop_diag = stop_diag
        self.indexing = util.diag_index_for_mat(n, start_diag, stop_diag)
        self.indexing_out = util.diag_index_for_mat(n, start_diag+1, stop_diag)

    def converter(self, signal:ArrayLike, seq:ArrayLike) -> ArrayLike:
        """Converts the input epigenetic signals into DLEM parameters.

        Args:
            signal (ArrayLike): epigenetic tracks

        Returns:
            ArrayLike: parameters, p_l, p_r
        """
        seq = self.seq_pooler(seq)
        layer_outs = []
        tmp = self.feature_batch_norm(torch.cat([signal, seq], axis=-2))
        for n, conv in enumerate(self.convs):
            tmp = conv(tmp)
            out_tmp = tmp[:,:self.channel_per_route]
            for trans_convs in self.trans_convs[-(n+1):]:
                out_tmp = trans_convs(out_tmp)
            layer_outs.append(out_tmp)
            tmp = tmp[:,self.channel_per_route:]
        out = torch.cat(layer_outs, axis=1)
        return self.mixer(out)

    def forward(self,
                diagonals:ArrayLike,
                signal:ArrayLike,
                seq:ArrayLike) -> ArrayLike:
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
        left_right = self.converter(signal.to(dev), seq.to(dev))
        out = torch.empty((diagonals.shape[0],
                           diagonals.shape[1]-(self.n-self.start_diag)), device="cpu")
        for index_diag in range(self.start_diag, self.stop_diag-1):
            diag_pred = torch.log(_dlem(torch.exp(diagonals[:, self.indexing(index_diag)]).to(dev),
                                        left_right, self.const, index_diag, self.n))
            diag_pred -=  torch.mean(diag_pred)
            out[:, self.indexing_out(index_diag+1)] = diag_pred.cpu()
        return out

    def contact_map_prediction(self,
                               signal:ArrayLike,
                               seq:ArrayLike,
                               init_mass:ArrayLike) -> ArrayLike:
        """Produce the contact map

        Args:
            init_mass (ArrayLike): initial mass for the algorithm to propagate. 

        Returns:
            ArrayLike: predicted contact mass.
        """
        dev = self.mixer[0].weight.device
        curr_diag = init_mass.to(dev)
        out_len = int((self.n - (self.stop_diag + self.start_diag - 1)/2) *
                      (self.stop_diag-self.start_diag))
        out_len -= (self.n - self.start_diag)
        with torch.no_grad():
            left_right = self.converter(signal.to(dev), seq.to(dev))
            out = torch.empty((init_mass.shape[0], out_len), device="cpu")
            for index_diag in range(self.start_diag, self.stop_diag-1):
                curr_diag = _dlem(curr_diag, left_right, self.const, index_diag, self.n)
                normed_diag = torch.log(curr_diag)
                normed_diag = normed_diag - normed_diag.mean(axis=1, keepdim=True)
                out[:, self.indexing_out(index_diag+1)] = normed_diag.cpu()
        return out

    def return_parameters(self, signal:ArrayLike, seq:ArrayLike) -> Tuple[ArrayLike,ArrayLike,ArrayLike]:
        """Return model parameters as a tuple.

        Returns:
            Tuple[ArrayLike,ArrayLike,ArrayLike]: parameters 
        """
        left_right = self.converter(signal, seq)
        left = left_right[:, 0, :]
        right = left_right[:, 1, :]
        return (left.detach().cpu().numpy(),
                right.detach().cpu().numpy())
