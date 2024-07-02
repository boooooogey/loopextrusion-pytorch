"""Implementation of DLEM with pytorch
"""
from typing import Tuple
from torch.nn import Module, Parameter, Sequential, ReLU, Sigmoid, Conv1d, ConvTranspose1d
import torch
from numpy.typing import ArrayLike

class DLEM(Module):
    """Predict contact map from Encode signals.
    """
    def __init__(self, n:int,
                       dim_num:int):
        """_summary_

        Args:
            left_init (ArrayLike): Initiation for the left parameters.
            right_init (ArrayLike): Initiation for the right parameters.
            free_unload (bool, optional): _description_. Defaults to False.
            type (int, optional): _description_. Defaults to 1.
        """
        super(DLEM, self).__init__()

        #self.converter = Sequential(
        #    Linear(in_features = dim_num, out_features = hidden_num),
        #    ReLU(),
        #    Linear(in_features = hidden_num, out_features = 2),
        #    Sigmoid()
        #)
        self.converter = Sequential(
            Conv1d(in_channels=dim_num, out_channels=10, kernel_size=3),
            ReLU(),
            Conv1d(in_channels=10, out_channels=10, kernel_size=1),
            ReLU(),
            ConvTranspose1d(in_channels=10, out_channels=10, kernel_size=3),
            ReLU(),
            Conv1d(in_channels=10, out_channels=2, kernel_size=1),
            Sigmoid()
        )
        #self.converter = Sequential(
        #    Conv1d(in_channels=dim_num, out_channels=5, kernel_size=3),
        #    ReLU(),
        #    #Conv1d(in_channels=10, out_channels=10, kernel_size=1),
        #    #ReLU(),
        #    ConvTranspose1d(in_channels=5, out_channels=2, kernel_size=3),
        #    #ReLU(),
        #    #Conv1d(in_channels=5, out_channels=2, kernel_size=1),
        #    Sigmoid()
        #)
        self.n = n
        #self.unload = Parameter(torch.ones(self.n) * unload_init, requires_grad=free_unload)
        self.const = Parameter(torch.tensor(0.99), requires_grad = True)

    def forward(self,
                signal:ArrayLike,
                curr_diag:ArrayLike,
                index_diag:int,
                transform:bool=True) -> ArrayLike:
        """forward operation for the network.

        Args:
            curr_diag (ArrayLike): current diagonal. current state.
            diag_i (int): diagonal index for the current state.
            transform (bool, optional): Should the output converted into log space and centered.
            Defaults to True.

        Returns:
            ArrayLike: prediction for the next state.
        """
        diag_len, n = curr_diag.shape[1], self.n

        left_right = self.converter(signal.transpose(-2,-1))
        left = left_right[:, 0, :]
        right = left_right[:, 1, :]
        #unload = left_right[:, 2, :]

        index_in_left = range(index_diag, n-1)
        index_in_right = range(1, n-index_diag)
        index_out_left = range(index_diag+1, n)
        index_out_right = range(0, n-index_diag-1)

        index_curr_diag_left = range(0, diag_len-1)
        index_curr_diag_right = range(1, diag_len)

        mass_in = curr_diag[:, index_curr_diag_right] * right[:, index_in_right]
        mass_in += curr_diag[:, index_curr_diag_left] * left[:, index_in_left]

        mass_out = right[:, index_out_right] + left[:, index_out_left]
        #mass_out += (1-self.unload[index_in_left]) * (1-self.unload[index_in_right])
        #mass_out += (1-unload[:, index_in_left]) * (1-unload[:, index_in_right])

        next_diag_pred = self.const * mass_in / mass_out

        if transform:
            next_diag_pred = torch.log(next_diag_pred)
            next_diag_pred = next_diag_pred - torch.mean(next_diag_pred)

        return next_diag_pred

    def contact_map_prediction(self, signal:ArrayLike, init_mass:ArrayLike) -> ArrayLike:
        """Produce the contact map

        Args:
            init_mass (ArrayLike): initial mass for the algorithm to propagate. 

        Returns:
            ArrayLike: predicted contact mass.
        """
        index_diag = self.n - init_mass.shape[1]
        out = torch.diag_embed(init_mass, offset=index_diag)
        curr_diag = init_mass
        with torch.no_grad():
            for diag in range(index_diag, self.n-1):
                curr_diag = self.forward(signal, curr_diag, diag, transform=False)
                out = out + torch.diag_embed(curr_diag, offset=diag+1)
        return out + torch.transpose(torch.triu(out, 1), -1,-2)

    #def project_to_constraints(self, lower:float, upper:float) -> None:
    #    """Project the parameters onto [lower, upper]

    #    Args:
    #        lower (float): region's lower bound
    #        upper (float): region's upper bound
    #    """
    #    with torch.no_grad():
    #        self.right.clamp_(lower, upper)
    #        self.left.clamp_(lower, upper)
    #        self.unload.clamp_(lower, upper)

    def return_parameters(self, signal:ArrayLike) -> Tuple[ArrayLike,ArrayLike,ArrayLike]:
        """Return model parameters as a tuple.

        Returns:
            Tuple[ArrayLike,ArrayLike,ArrayLike]: parameters 
        """
        left_right = self.converter(signal)
        left = left_right[:, :, 0]
        right = left_right[:, :, 1]
        return (left.detach().cpu().numpy(),
                right.detach().cpu().numpy())
