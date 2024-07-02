"""Implementation of DLEM with pytorch
"""
from typing import Tuple
from torch.nn import Module
from torch.nn import Parameter
import torch
from numpy.typing import ArrayLike

class DLEM(Module):
    """Differentiable loop extrusion model in pytorch.
    """
    def __init__(self, left_init:ArrayLike,
                       right_init:ArrayLike,
                       unload_init:float=0.95,
                       free_unload:bool=False):
        """_summary_

        Args:
            left_init (ArrayLike): Initiation for the left parameters.
            right_init (ArrayLike): Initiation for the right parameters.
            free_unload (bool, optional): _description_. Defaults to False.
            type (int, optional): _description_. Defaults to 1.
        """
        left_init, right_init = torch.Tensor(left_init), torch.Tensor(right_init)
        super(DLEM, self).__init__()
        self.const = Parameter(torch.tensor(0.99), requires_grad = True)
        self.n = len(left_init)
        self.left = Parameter(left_init, requires_grad=True)
        self.right = Parameter(right_init, requires_grad=True)
        self.unload = Parameter(torch.ones_like(left_init) * unload_init, requires_grad=free_unload)

    def forward(self,
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
        #curr_diag = torch.exp(curr_diag)
        diag_len, n = curr_diag.shape[-1], self.n

        index_in_left = range(index_diag, n-1)
        index_in_right = range(1, n-index_diag)
        index_out_left = range(index_diag+1, n)
        index_out_right = range(0, n-index_diag-1)

        index_curr_diag_left = range(0, diag_len-1)
        index_curr_diag_right = range(1, diag_len)

        mass_in = curr_diag[:, index_curr_diag_right] * self.right[index_in_right]
        mass_in += curr_diag[:, index_curr_diag_left] * self.left[index_in_left]

        mass_out = self.right[index_out_right] + self.left[index_out_left]
        mass_out += (1-self.unload[index_in_left]) * (1-self.unload[index_in_right])

        next_diag_pred = self.const * mass_in / mass_out.reshape(1,-1)

        if transform:
            next_diag_pred = torch.log(next_diag_pred)
            next_diag_pred = next_diag_pred - torch.mean(next_diag_pred)

        return next_diag_pred

    def contact_map_prediction(self, init_mass:ArrayLike) -> ArrayLike:
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
                curr_diag = self.forward(curr_diag, diag, transform=False)
                out = out + torch.diag_embed(curr_diag, offset=diag+1)
        return out +torch.transpose(torch.triu(out, 1), -1,-2)

    def project_to_constraints(self, lower:float, upper:float) -> None:
        """Project the parameters onto [lower, upper]

        Args:
            lower (float): region's lower bound
            upper (float): region's upper bound
        """
        with torch.no_grad():
            self.right.clamp_(lower, upper)
            self.left.clamp_(lower, upper)
            self.unload.clamp_(lower, upper)

    def return_parameters(self) -> Tuple[ArrayLike,ArrayLike,ArrayLike]:
        """Return model parameters as a tuple.

        Returns:
            Tuple[ArrayLike,ArrayLike,ArrayLike]: parameters 
        """
        return (self.left.detach().cpu().numpy(),
                self.right.detach().cpu().numpy(),
                self.unload.detach().cpu().numpy())
