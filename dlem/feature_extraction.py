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

def train_extractor(patch:ArrayLike,
                    diag_stop:int,
                    learning_rate:float,
                    arch:str,
                    diag_start:int,
                    loss:torch.nn.Module,
                    patience:int,
                    num_epoch:int,
                    dev_name:str,
                    **kwargs:Any) -> Tuple[torch.nn.Module,torch.nn.Module,np.ndarray,np.ndarray]:
    """Train the feature extractor model and return the best models and loss arrays.

    Args:
        patch (ArrayLike): The input patch representing the contact map.
        diag_stop (int): The stopping diagonal index for the dlem algorithm.
        learning_rate (float): The learning rate for the optimization algorithm.
        arch (str): The architecture of the dlem model.
        diag_start (int): The starting diagonal index for the dlem algorithm.
        loss (torch.nn.Module): The loss function for the optimization algorithm.
        patience (int): The number of epochs to wait for improvement in the validation loss before
        reducing the learning rate.
        num_epoch (int): The number of epochs to train the model.
        dev_name (str): The name of the device to use for computation.

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module, np.ndarray, np.ndarray]: A tuple containing the best
        loss model, best correlation model, loss array, and correlation array.
    """
    dev = torch.device(dev_name)
    architecture = load_model(arch)
    model = architecture(patch.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, mode="max")
    best_loss_model, best_corr_model, arr_loss, arr_corr = util.train(
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
    return best_loss_model, best_corr_model, arr_loss, arr_corr

def plot_model(patch:ArrayLike,
               best_corr_model:torch.nn.Module,
               dev_name:str,
               diag_start:int,
               diag_stop:int,
               params:Tuple[ArrayLike, ArrayLike],
               plot_path:str) -> None:
    """Plot the contact map prediction and save the plot.

    Args:
        patch (ArrayLike): The input patch representing the contact map.
        best_corr_model (torch.nn.Module): The best correlation model obtained from training.
        dev_name (str): The name of the device to use for computation.
        diag_start (int): The starting diagonal index for the dlem algorithm.
        diag_stop (int): The stopping diagonal index for the dlem algorithm.
        params (Tuple[ArrayLike, ArrayLike]): The parameters of the best correlation model.
        plot_path (str): The path to save the plot.
    """
    dev = torch.device(dev_name)
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

def extractor(patch:ArrayLike,
              diag_stop:int,
              learning_rate:float=0.5,
              arch:str="netdlem2",
              diag_start:int=1,
              loss:Union[torch.nn.Module,None]=None,
              patience:int=25,
              num_epoch:int=100,
              dev_name:str='cuda',
              do_plot:bool=False,
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
    if loss is None:
        loss = torch.nn.MSELoss(reduction='mean')
    if do_plot and (plot_path is None):
        raise ValueError("Plotting turned on however path to save is not provided.")
    _, best_corr_model, _, arr_corr =train_extractor(patch,
                                                     diag_stop,
                                                     learning_rate,
                                                     arch,
                                                     diag_start,
                                                     loss,
                                                     patience,
                                                     num_epoch,
                                                     dev_name,
                                                     **kwargs)
    params = best_corr_model.return_parameters()
    if do_plot:
        plot_model(patch, best_corr_model, dev_name, diag_start, diag_stop, params, plot_path)
    return params, np.max(arr_corr)
