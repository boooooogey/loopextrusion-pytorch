
from typing import Tuple, List
from torch.nn import Module
from torch.nn import Parameter
import torch
from numpy.typing import ArrayLike

class DLEM(Module):
    """Differentiable loop extrusion model in pytorch.
    """


    def __init__(self, 
                 n:int,   # window size
                 res: int,  # resolution
                 left_init:ArrayLike=None,
                 right_init:ArrayLike=None,
                 detach:float=None
                ):
        
        """Initialize DLEM model.
    
        Args:
            n: int, window size
            res: int, resolution
            left_init: blocks left leg initialization values
            right_init: blocks right leg initialization values
            detach: scalar detach rate
        """
        super(DLEM, self).__init__()

        if left_init == None:
            self.left = Parameter(torch.ones(n) * 0.99, requires_grad=True)
            self.right = Parameter(torch.ones(n) * 0.99, requires_grad=True)
            
        else:
            left_init, right_init = torch.Tensor(left_init), torch.Tensor(right_init)
            self.left = Parameter(left_init, requires_grad=True)
            self.right = Parameter(right_init, requires_grad=True)
        
        if detach == None:
            res_detach = {10000:0.025,
                           5000:0.0125,
                           2000:0.005}
            self.detach = torch.Tensor([res_detach[res]])

        else:
            self.detach = torch.Tensor([detach])    
        
        self.n = n

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

        index_in_right = range(index_diag, n-1)   # j
        index_in_left = range(1, n-index_diag)  # i
        index_out_right = range(index_diag+1, n)  # j+1
        index_out_left = range(0, n-index_diag-1) # i-1 
        
        index_curr_diag_right = range(0, diag_len-1)
        index_curr_diag_left = range(1, diag_len)

        mass_in = curr_diag[:, index_curr_diag_right] * self.right[index_in_right]
        mass_in += curr_diag[:, index_curr_diag_left] * self.left[index_in_left]

        mass_out = self.right[index_out_right] + self.left[index_out_left]

        mass_out += self.detach[0]
    
        # next_diag_pred = self.const * mass_in / mass_out.reshape(1,-1)
        next_diag_pred = mass_in / mass_out.reshape(1,-1)

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
            self.right.clamp_(0.05, upper)
            self.left.clamp_(0.05, upper)
            

    def return_parameters(self) -> Tuple[ArrayLike,ArrayLike,ArrayLike]:
        """Return model parameters as a tuple.

        Returns:
            Tuple[ArrayLike,ArrayLike,ArrayLike]: parameters 
        """
            
        return (self.left.detach().cpu().numpy(),
                self.right.detach().cpu().numpy()           
                )


    def return_parameter_names(self) -> List[str]:
            """Return the parameter names.

            Returns:
                List[str]: parameter names
            """
            return ["left", "right"]

