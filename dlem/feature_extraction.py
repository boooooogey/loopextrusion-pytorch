"""Fit dlem and return one-dimensional features
"""
from typing import Tuple, Union
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
from numpy.typing import ArrayLike
from dlem import util
from dlem import load_model

def extractor(patch:ArrayLike,
              learning_rate:float=0.5,
              arch:str="netdlem2",
              diag_start:int=3,
              diag_stop:Union[int,None]=None,
              loss:Union[torch.nn.Module,None]=None,
              patience:int=25,
              dev_name:str='cuda',
              do_plot:bool=False,
              plot_path:Union[str,None]=None) -> Tuple[Tuple[ArrayLike,ArrayLike],float]:
    """Extract dlem one dimensional features from contactmaps.

    Args:
        patch (ArrayLike): _description_
        learning_rate (float, optional): _description_. Defaults to 0.5.
        arch (str, optional): _description_. Defaults to "netdlem2".
        diag_start (int, optional): _description_. Defaults to 3.
        diag_stop (Union[int,None], optional): _description_. Defaults to None.
        loss (Union[torch.Module,None], optional): _description_. Defaults to None.
        patience (int, optional): _description_. Defaults to 25.
        dev_name (str, optional): _description_. Defaults to 'cuda'.
        do_plot (bool, optional): _description_. Defaults to False.
        plot_path (Union[str,None], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        Tuple[Tuple[ArrayLike,ArrayLike],float]: _description_
    """

    if diag_stop is None:
        diag_stop = int(np.floor(patch.shape[0]*0.3))
    if loss is None:
        loss = torch.nn.MSELoss(reduction='mean')
    if do_plot and (plot_path is None):
        raise ValueError("Plotting turned on however path to save is not provided.")
    dev = torch.device(dev_name)
    architecture = load_model(arch)
    model = architecture(np.ones(patch.shape[0]) * 0.95, np.ones(patch.shape[0]) * 0.95)
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
        dev,
        num_epoch=100)
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
    return params, np.max(arr_corr)
