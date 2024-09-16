"""Fit dlem and return one-dimensional features
"""
from typing import Tuple, Union, Any
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
from numpy.typing import ArrayLike
from . import util
from . import load_model

def extractor(patch:ArrayLike,
              learning_rate:float=0.5,
              arch:str="netdlem2",
              diag_start:int=3,
              diag_stop:Union[int,None]=None,
              loss:Union[torch.nn.Module,None]=None,
              patience:int=25,
              num_epoch:int=100,
              dev_name:str='cuda',
              do_plot:bool=False,
              return_best_corr_pred:bool=False,
              plot_path:Union[str,None]=None,
              **kwargs:Any) -> Tuple[Tuple[ArrayLike,ArrayLike],float]:
    """Extract dlem one-dimensional features from contact maps.

    This function takes a contact map represented as a patch and extracts one-dimensional features
    using the dlem algorithm.

    Args:
        patch (ArrayLike): The contact map represented as a patch.
        learning_rate (float, optional): The learning rate for the optimization algorithm. Defaults
        to 0.5.
        arch (str, optional): The architecture of the dlem model. Defaults to "netdlem2".
        diag_start (int, optional): The starting diagonal index for the dlem algorithm. Defaults to
        3.
        diag_stop (Union[int,None], optional): The stopping diagonal index for the dlem algorithm.
        Defaults to None.
        loss (Union[torch.Module,None], optional): The loss function for the optimization algorithm.
        Defaults to None.
        patience (int, optional): The number of epochs to wait for improvement in the validation
        loss before reducing the learning rate. Defaults to 25.
        dev_name (str, optional): The name of the device to use for computation. Defaults to 'cuda'.
        do_plot (bool, optional): Whether to plot the results. Defaults to False.
        plot_path (Union[str,None], optional): The path to save the plot. Defaults to None.

    Raises:
        ValueError: If plotting is turned on but no plot path is provided.

    Returns:
        Tuple[Tuple[ArrayLike,ArrayLike],float]: A tuple containing the extracted features and the
        maximum correlation value.
    """

    if diag_stop is None:
        diag_stop = int(np.floor(patch.shape[0]*0.3))
    if loss is None:
        loss = torch.nn.MSELoss(reduction='mean')
    if do_plot and (plot_path is None):
        raise ValueError("Plotting turned on however path to save is not provided.")
    dev = torch.device(dev_name)
    architecture = load_model(arch)
    model = architecture(patch.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, mode="max")
    _, best_corr_model, _, arr_corr = util.train(
        model,
        optimizer,
        scheduler,
        loss,
        np.exp(patch)[np.newaxis],
        diag_start,
        diag_stop,
        dev=dev,
        num_epoch=num_epoch,
        **kwargs)
    params = best_corr_model.return_parameters()
    if do_plot:
        best_corr_pred = best_corr_model.contact_map_prediction(
            torch.ones((1, patch.shape[0]), device=dev) * patch.shape[0]
        ).detach().cpu().numpy()
        best_corr_pred = util.diagonal_normalize(np.log(best_corr_pred))
        best_corr_pred = best_corr_pred[0]
        util.plot_results(patch,
                          best_corr_pred,
                          params,
                          ignore_i=diag_start,
                          ignore_i_off=diag_stop,
                          cmap="vlag")
        plt.savefig(plot_path)
        plt.close()
    if return_best_corr_pred:
        best_corr_pred = best_corr_model.contact_map_prediction(
            torch.ones((1, patch.shape[0]), device=dev) * patch.shape[0]
        ).detach().cpu().numpy()
        best_corr_pred = util.diagonal_normalize(np.log(best_corr_pred))
        best_corr_pred = best_corr_pred[0]
        return params, np.max(arr_corr), best_corr_pred
    return params, np.max(arr_corr)
