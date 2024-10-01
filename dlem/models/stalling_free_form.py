"""Implementation of DLEM with option for stalling

Changes wrt netdlem2:
    
Models
c[i, j+1] =  (pR[j] * c[i,j]  + pL[i+1] * c[i+1,j+1] ) / ( pR[j+1] + pL[i]  + lm +  s*(1 - pL[j+1] + 1 -pR[i])     
instead of
c[i, j+1] =  (pR[j] * c[i,j]  + pL[i+1] * c[i+1,j+1] ) / ( pR[j+1] + pL[i]  + lm[i] +  lm[j+1] )                  # or lm[i] *  lm[j+1]


to vary the stalling constant s, the module has additional parameter stall and two additional inputs: stall_input and free_stall. 

Forward:
there are additional index variables in the forward model:
index_stall_left = range(0, n-index_diag-1)
index_stall_right = range(index_diag+1, n)                                  

and additional term in the mass outflux:
mass_out += self.stall * (1 - self.right[index_stall_right] + 1 - self.left[index_stall_left]) 

the "spotnaneous offloading" is also converted to a constant from a vector:
mass_out +=  1 - self.const

self.const is instead removed from the final expression mass_in / mass_out # * self.const     


Reflection:
reflection for stall added 

    
"""
from typing import Tuple, List
from torch.nn import Module
from torch.nn import Parameter
import torch
from numpy.typing import ArrayLike

class DLEM(Module):
    """Differentiable loop extrusion model in pytorch.
    """
    def __init__(self, n,
                       unload_init:float=0.95,
                       free_unload:bool=False,
                       free_stall:bool=False,
                       stall_init:float=0.0):   # stalling constant
        """_summary_

        Args:
            left_init (ArrayLike): Initiation for the left parameters.
            right_init (ArrayLike): Initiation for the right parameters.
            free_unload (bool, optional): _description_. Defaults to False.
            type (int, optional): _description_. Defaults to 1.
        """
        super(DLEM, self).__init__()
        self.const = Parameter(torch.tensor(1.00), requires_grad = True)
        self.n = n
        self.left = Parameter(torch.ones(n) * 0.95, requires_grad=True)
        self.right = Parameter(torch.ones(n) * 0.95, requires_grad=True)
        self.unload = Parameter(torch.ones(n) * unload_init, requires_grad=free_unload)

        self.stalling = torch.nn.Sequential(torch.nn.Linear(in_features=2, out_features=10),
                                            torch.nn.Tanh(),
                                            torch.nn.Linear(in_features=10, out_features=1),
                                            torch.nn.Sigmoid())

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
        diag_len, n = curr_diag.shape[-1], self.n

        index_in_left = range(index_diag, n-1)
        index_in_right = range(1, n-index_diag)
        index_out_left = range(index_diag+1, n)
        index_out_right = range(0, n-index_diag-1)

        index_stall_left = range(0, n-index_diag-1)
        index_stall_right = range(index_diag+1, n)

        ####
        index_curr_diag_left = range(0, diag_len-1)
        index_curr_diag_right = range(1, diag_len)

        mass_in = curr_diag[:, index_curr_diag_right] * self.right[index_in_right]
        mass_in += curr_diag[:, index_curr_diag_left] * self.left[index_in_left]

        mass_out = self.right[index_out_right] + self.left[index_out_left]
        #mass_out +=  1 - self.const

        #Stalling term added here
        stall = self.stalling(
            torch.cat((self.right[index_stall_right].reshape(-1,1),
                       self.left[index_stall_left].reshape(-1,1)), dim=1)).reshape(-1)
        mass_out += self.const * stall
        #mass_out += self.stall * (1 - self.right[index_stall_right] + 1 - self.left[index_stall_left])

        next_diag_pred = mass_in / mass_out

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
        return out + torch.transpose(torch.triu(out, 1), -1,-2)

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
            self.const.clamp_(lower, upper)

    def return_parameters(self) -> Tuple[ArrayLike,ArrayLike]:
        """Return model parameters as a tuple.

        Returns:
            Tuple[ArrayLike,ArrayLike,ArrayLike]: parameters 
        """
        return (self.left.detach().cpu().numpy(),
                self.right.detach().cpu().numpy())

    def return_parameter_names(self) -> List[str]:
        """Return the parameter names.

        Returns:
            List[str]: parameter names
        """
        return ["left", "right"]

    def return_stalling(self, index_diag:int) -> ArrayLike:
                
        n = self.n

        index_stall_left = range(0, n-index_diag-1)
        index_stall_right = range(index_diag+1, n)

        #Stalling term added here
        stall = self.stalling(
            torch.cat((self.right[index_stall_right].reshape(-1,1),
                       self.left[index_stall_left].reshape(-1,1)), dim=1)).reshape(-1)


        return stall