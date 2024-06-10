"""Utility functions for DLEM training and visualization
"""
from typing import Any, Tuple, Union
import copy
from torch.nn import Module
from torch.nn.functional import avg_pool2d
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import device
from torch import Tensor
import torch
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def tile_patch(patch:ArrayLike, kernel_size:int):
    """Pools the average from each tile size of which are determined using kernel_size.

    Args:
        patch (ArrayLike): input matrix.
        kernel_size (int): edge length of the square tile.

    Returns:
        ArrayLike: tiled smaller matrix.
    """
    return avg_pool2d(torch.tensor(patch).unsqueeze(0),
                      kernel_size=kernel_size,
                      stride=kernel_size)[0]

def mat_corr(mat1:Tensor, mat2:Tensor) -> Tensor:
    """Correlation between two matrices.

    Args:
        mat1 (Tensor): input 1.
        mat2 (Tensor): input 2. 

    Returns:
        Tensor: correlation
    """
    return torch.corrcoef(torch.vstack([mat1.flatten(), mat2.flatten()]))[0,1]

def vec_corr(vec1:Tensor, vec2:Tensor) -> Tensor:
    """Correlation between two vectors.

    Args:
        vec1 (Tensor): input 1.
        vec2 (Tensor): input 2. 

    Returns:
        Tensor: correlation
    """
    return torch.corrcoef(torch.vstack([vec1, vec2]))[0,1]

def ignore_diag_plot(mat:ArrayLike, num_diag:int) -> ArrayLike:
    """ignore first few diagonals for plotting purposes.

    Args:
        mat (ArrayLike): input patch 
        num_diag (int): ignores this number of diagonals.

    Returns:
        ArrayLike: masked patch 
    """
    n = mat.shape[0]
    return mat + np.tril(np.triu(np.full((n, n), -np.inf), -num_diag), num_diag)

def diagonal_normalize(mat:ArrayLike) -> ArrayLike:
    """Center each diagonal mean at 0.

    Args:
        mat (ArrayLike): input patch.

    Returns:
        ArrayLike: matrix with 0 centered diagonals.
    """
    out = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        diag_i = np.diag(mat, i)
        out += np.diag(diag_i - np.mean(diag_i), i)
    return out + np.triu(out, 1).T

def train(model:Module,
          optimizer:Optimizer,
          scheduler:LRScheduler,
          loss:Module,
          data:ArrayLike,
          diag_start:int,
          diag_end:int,
          dev:device=None,
          num_epoch:int=100,
          parameter_lower_bound:float=1e-9,
          parameter_upper_bound:float=1.0) -> Tuple[Module, Module, ArrayLike, ArrayLike]:
    """Train diagonal model.

    Args:
        model (Module): DLEM type of model
        optimizer (Optimizer): any optimizer. For example ADAM.
        scheduler (LRScheduler): learning rate scheduler. For example plateau scheduler.
        loss (Module): a loss function. For example MSE.
        data (ArrayLike): patch.
        diag_start (int): starting diagonal index.
        diag_end (int): stopping diagonal index.
        dev (device, optional): device if None then gpu if cuda is avaible, cpu otherwise.
        Defaults to None.
        num_epoch (int, optional): Number of epochs. Defaults to 100.
        parameter_lower_bound (float, optional): constrain the parameters. Lower limit.
        Defaults to 1e-5.
        parameter_upper_bound (float, optional): constrain the parameters. Upper limit.
        Defaults to 1.0.

    Returns:
        (Module, Module, ArrayLike, ArrayLike) : the best model in terms of loss
                                                 the best model in terms of correlation
                                                 the trajectory of loss
                                                 the trajectory of correlation
    """

    data = torch.tensor(data).to(dev)
    init_diag = torch.ones(data.shape[0]) * data.shape[0]
    init_diag = init_diag.to(dev)
    best_corr = 0
    arr_corr = []
    best_loss = torch.inf
    arr_loss = []
    model = model.to(dev)

    mask = torch.ones_like(data, dtype=bool)
    train_ii = torch.where(
        torch.tril(torch.triu(mask, diag_start),
                   diag_end) + torch.triu(torch.tril(mask, -diag_start), -diag_end)
    )

    for e in range(num_epoch):
        optimizer.zero_grad()
        loss_total = 0
        pred_map = model.contact_map_prediction(init_diag)
        curr_cor = mat_corr(pred_map[train_ii], data[train_ii])
        arr_corr.append(curr_cor.detach().cpu().numpy())
        for diag_i in range(diag_start, diag_end):
            pred = model(data.diag(diag_i), diag_i, True)
            loss_total += loss(pred, torch.log(data.diag(diag_i+1)))
        if loss_total < best_loss:
            best_loss = loss_total
            best_loss_model = copy.deepcopy(model)
        if curr_cor > best_corr:
            best_corr = curr_cor
            best_corr_model = copy.deepcopy(model)
        loss_total.backward()
        optimizer.step()
        arr_loss.append(loss_total.detach().cpu().numpy())
        scheduler.step(curr_cor)

        model.project_to_constraints(parameter_lower_bound, parameter_upper_bound)
        print(f'{int((e+1)/num_epoch*100):3}/100: '
              f'correlation = {curr_cor:.3f}, '
              f'loss = {loss_total:.3f}',
              flush=True, end='\r')

    return best_loss_model, best_corr_model, np.array(arr_loss), np.array(arr_corr)

def plot_pred_data_on_same_ax(mat:ArrayLike, diag_ignore:int, diag_ignore_off:int, ax:Any):
    """Plot prediction and data on the same ax but with different color schemes.

    Args:
        mat (ArrayLike): input matrix with upper triangle and lower triangle are data and
                         prediction.
        diag_ignore (int): ignore diagonals until.
        ax (Any): pyplot axes.
    """

    # Create masks for plotting
    mask_upper = np.tril(np.triu(np.ones_like(mat, dtype=bool), diag_ignore), diag_ignore_off-1)
    mask_lower = np.triu(np.tril(np.ones_like(mat, dtype=bool), -diag_ignore), -diag_ignore_off+1)

    mat_upper = np.ma.masked_where(~mask_upper, mat)
    mat_lower = np.ma.masked_where(~mask_lower, mat)

    # Create a custom colormap and normalization
    norm_upper = Normalize(vmin=mat_upper.min(), vmax=mat_upper.max())
    norm_lower = Normalize(vmin=mat_lower.min(), vmax=mat_lower.max())

    # Plot the matrix
    ax.imshow(mat_upper, cmap="icefire", norm=norm_upper)
    ax.imshow(mat_lower, cmap="icefire", norm=norm_lower)


def plot_results(patch:ArrayLike, pred:ArrayLike,
                 params:Tuple, axes:Any=None, ignore_i:int=3, ignore_i_off:Union[int,None]=None,
                 start:Union[int,None]=None, end:Union[int,None]=None):
    """Plot results of the fit.

    Args:
        start (int): start index
        end (int): end index 
        patch (ArrayLike): input data
        pred (ArrayLike): model prediction 
        params (Tuple): parameters of the model. 
        axes (Any, optional): can be used to integrate into a bigger image. Defaults to None.
        ignore_i (int): ignore diagonals until ith one. Defaults to 3.
    """
    if start is None:
        start = 0
    if end is None:
        end = patch.shape[0]
    if ignore_i_off is None:
        ignore_i_off = int(np.floor(patch.shape[0]*0.3))

    plot_mat = np.triu(patch) + np.triu(pred, 1).T
    if axes is None:
        _, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,14),
                               sharex='col',
                               sharey='row',
                               gridspec_kw={'wspace':0,
                                           'hspace':0,
                                           'height_ratios':[10, 50, 10],
                                           'width_ratios':[50, 10]})
    axes[0, 1].remove()
    axes[2, 1].remove()
    axes[0, 0].plot(np.arange(end-start), params[1][start:end])
    #axes[1, 0].matshow(plot_mat[start:end, start:end], cmap="icefire")
    plot_pred_data_on_same_ax(plot_mat, ignore_i, ignore_i_off, axes[1, 0])
    axes[1, 1].plot(params[0][start:end], np.arange(end-start))
    axes[2, 0].plot(np.arange(end-start), params[2][start:end])

def train_w_signal(model:Module,
                   optimizer:Optimizer,
                   scheduler:LRScheduler,
                   loss:Module,
                   data:ArrayLike,
                   sig:ArrayLike,
                   diag_start:int,
                   diag_end:int,
                   dev:device=None,
                   num_epoch:int=100) -> Tuple[Module, Module, ArrayLike, ArrayLike]:
    """Train diagonal model.

    Args:
        model (Module): DLEM type of model
        optimizer (Optimizer): any optimizer. For example ADAM.
        scheduler (LRScheduler): learning rate scheduler. For example plateau scheduler.
        loss (Module): a loss function. For example MSE.
        data (ArrayLike): patch.
        diag_start (int): starting diagonal index.
        diag_end (int): stopping diagonal index.
        dev (device, optional): device if None then gpu if cuda is avaible, cpu otherwise.
        Defaults to None.
        num_epoch (int, optional): Number of epochs. Defaults to 100.
        parameter_lower_bound (float, optional): constrain the parameters. Lower limit.
        Defaults to 1e-5.
        parameter_upper_bound (float, optional): constrain the parameters. Upper limit.
        Defaults to 1.0.

    Returns:
        (Module, Module, ArrayLike, ArrayLike) : the best model in terms of loss
                                                 the best model in terms of correlation
                                                 the trajectory of loss
                                                 the trajectory of correlation
    """

    def get_diags(mat, i):
        return torch.diagonal(mat,i,dim1=-2,dim2=-1)

    data = torch.tensor(data).to(dev)
    init_diag = torch.ones((1, data.shape[1])) * data.shape[1]
    init_diag = init_diag.to(dev)
    best_corr = 0
    arr_corr = []
    best_loss = torch.inf
    arr_loss = []
    model = model.to(dev)
    sig = sig.to(dev)

    mask = torch.ones_like(data, dtype=bool)
    train_ii = torch.where(
        torch.tril(torch.triu(mask, diag_start),
                   diag_end) + torch.triu(torch.tril(mask, -diag_start), -diag_end)
    )

    for e in range(num_epoch):
        optimizer.zero_grad()
        loss_total = 0
        pred_map = model.contact_map_prediction(sig, init_diag)
        curr_cor = mat_corr(pred_map[train_ii], data[train_ii])
        arr_corr.append(curr_cor.detach().cpu().numpy())
        for diag_i in range(diag_start, diag_end):
            pred = model(sig, get_diags(data, diag_i), diag_i, True)
            loss_total += loss(pred, torch.log(get_diags(data, diag_i+1)))
        if loss_total < best_loss:
            best_loss = loss_total
            best_loss_model = copy.deepcopy(model)
        if curr_cor > best_corr:
            best_corr = curr_cor
            best_corr_model = copy.deepcopy(model)
        loss_total.backward()
        optimizer.step()
        arr_loss.append(loss_total.detach().cpu().numpy())
        scheduler.step(curr_cor)

        print(f'{int((e+1)/num_epoch*100):3}/100: '
              f'correlation = {curr_cor:.3f}, '
              f'loss = {loss_total:.3f}',
              flush=True, end='\r')

    return best_loss_model, best_corr_model, np.array(arr_loss), np.array(arr_corr)
