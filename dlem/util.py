"""Utility functions for DLEM training and visualization
"""
from typing import Any, Tuple, Union, List
import copy
import json
import pandas as pd
from torch.nn import Module
import torch.nn.functional as func
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import device
from torch import Tensor
import torch
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn
from IPython.display import display, clear_output


def get_diags(mat:ArrayLike, i:int) -> ArrayLike:
    """return diagonal from batched matrices.

    Args:
        mat (ArrayLike): batch of matrices
        i (int): ith diagonal

    Returns:
        ArrayLike: list of diagonals
    """
    return torch.diagonal(mat,i,dim1=-2,dim2=-1)

def tile_patch(patch:ArrayLike, kernel_size:int):
    """Pools the average from each tile size of which are determined using kernel_size.

    Args:
        patch (ArrayLike): input matrix.
        kernel_size (int): edge length of the square tile.

    Returns:
        ArrayLike: tiled smaller matrix.
    """
    if isinstance(patch, np.ndarray):
        patch = torch.from_numpy(patch)
    return func.avg_pool2d(patch.unsqueeze(0),
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

def vec_corr_batch(vec1:Tensor, vec2:Tensor, collapse:callable=None) -> Tensor:
    """Correlation between of two batches. The shape is assumed to be 2 dimensional. The first
    dimension is the batch size and the second dimension is the vector lenght. The corresponding
    vectors are also assumed to be in the same order.

    Args:
        vec1 (Tensor): input 1.
        vec2 (Tensor): input 2. 
        collapse (callable): function to collapse the correlation. Defaults to torch.mean().

    Returns:
        Tensor: correlation
    """
    assert vec1.shape[0] == vec2.shape[0]
    n = vec1.shape[0]
    if collapse is None:
        collapse = torch.mean
    return collapse(torch.corrcoef(torch.vstack([vec1, vec2]))[range(n),range(n, 2*n)])

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
    if isinstance(mat, np.ndarray):
        mat = torch.from_numpy(mat)
    out = torch.zeros_like(mat)
    for i in range(mat.shape[1]):
        diag_i = get_diags(mat, i)
        out += torch.diag_embed(diag_i - torch.mean(diag_i, axis=1)[:,None], i)
    return out + torch.triu(out, 1).transpose(-2,-1)

def diagonal_region_indices_from(mat:ArrayLike,
                                 start:int,
                                 stop:int) -> List[ArrayLike]:
    """Return indices for the diagonal region.

    Args:
        mat (ArrayLike): mat that indices are going to be used for.
        start (int): start diagonal.
        stop (int): stop diagonal 

    Returns:
        Tuple[ArrayLike, ArrayLike]: the indices for every dimension of the array.
    """
    mask = torch.ones_like(mat, dtype=bool)
    return torch.where(
                torch.tril(torch.triu(mask, start),
                stop) + torch.triu(torch.tril(mask, -start), -stop)
           )

def train(model:Module,
          optimizer:Optimizer,
          scheduler:LRScheduler,
          loss:Module,
          data:ArrayLike,
          diag_start:int,
          diag_end:int,
          do_print:bool=False,
          dev:device=None,
          num_epoch:int=100,
          parameter_lower_bound:float=1e-9,
          parameter_upper_bound:float=1.0,
          early_thresh:float=1e-5) -> Tuple[Module, Module, ArrayLike, ArrayLike]:
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

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.to(dev)
    init_diag = torch.ones((data.shape[0], data.shape[1])) * data.shape[1]
    init_diag = init_diag.to(dev)
    best_corr = -torch.inf
    arr_corr = []
    best_loss = torch.inf
    arr_loss = []
    model = model.to(dev)

    train_ii = diagonal_region_indices_from(data, diag_start, diag_end)

    best_loss_model, best_corr_model = copy.deepcopy(model), copy.deepcopy(model)

    for e in range(num_epoch):
        curr_lr = scheduler.get_last_lr()[-1]
        if curr_lr < early_thresh:
            break
        optimizer.zero_grad()
        loss_total = 0
        pred_map = model.contact_map_prediction(init_diag)
        pred_map = torch.exp(diagonal_normalize(torch.log(pred_map)))
        curr_cor = mat_corr(pred_map[train_ii], data[train_ii])
        arr_corr.append(curr_cor.detach().cpu().numpy())
        for diag_i in range(diag_start, diag_end):
            pred = model(get_diags(data, diag_i), diag_i, True)
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

        model.project_to_constraints(parameter_lower_bound, parameter_upper_bound)
        if do_print:
            print(f'{int((e+1)/num_epoch*100):3}/100: '
                f'correlation = {curr_cor:.3f}, '
                f'loss = {loss_total:.3f}',
                flush=True, end='\r')

    return best_loss_model, best_corr_model, np.array(arr_loss), np.array(arr_corr)

def plot_pred_data_on_same_ax(mat:ArrayLike,
                              diag_ignore:int,
                              diag_ignore_off:int,
                              ax:Any, cmap=None):
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
    ax.imshow(mat_upper, cmap=cmap, norm=norm_upper)
    ax.imshow(mat_lower, cmap=cmap, norm=norm_lower)


def plot_results(patch:ArrayLike, pred:ArrayLike,
                 params:Tuple, axes:Any=None, ignore_i:int=3, ignore_i_off:Union[int,None]=None,
                 start:Union[int,None]=None, end:Union[int,None]=None, cmap=None,
                 scale:float=1.0):
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
        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12*scale,12*scale),
                               sharex='col',
                               sharey='row',
                               gridspec_kw={'wspace':0,
                                           'hspace':0,
                                           'height_ratios':[1, 5],
                                           'width_ratios':[5, 1]})
    axes[0, 1].remove()
    #axes[2, 1].remove()
    axes[0, 0].plot(np.arange(end-start), params[1][start:end])
    #axes[1, 0].matshow(plot_mat[start:end, start:end], cmap="icefire")
    plot_pred_data_on_same_ax(plot_mat, ignore_i, ignore_i_off, axes[1, 0], cmap=cmap)
    axes[1, 1].plot(params[0][start:end], np.arange(end-start))
    #axes[2, 0].plot(np.arange(end-start), params[2][start:end])

def train_w_signal(model:Module,
                   optimizer:Optimizer,
                   scheduler:LRScheduler,
                   loss:Module,
                   data:ArrayLike,
                   sig:ArrayLike,
                   diag_start:int,
                   diag_end:int,
                   dev:device=None,
                   num_epoch:int=100,
                   early_thresh:float=1e-5) -> Tuple[Module, Module, ArrayLike, ArrayLike]:
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

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.to(dev)
    init_diag = torch.ones((1, data.shape[1])) * data.shape[1]
    init_diag = init_diag.to(dev)
    best_corr = -torch.inf
    arr_corr = []
    best_loss = torch.inf
    arr_loss = []
    model = model.to(dev)
    sig = sig.to(dev)

    train_ii = diagonal_region_indices_from(data, diag_start, diag_end)

    for e in range(num_epoch):
        curr_lr = scheduler.get_last_lr()[-1]
        if curr_lr < early_thresh:
            break
        optimizer.zero_grad()
        loss_total = 0
        pred_map = model.contact_map_prediction(sig, init_diag)
        pred_map = torch.exp(diagonal_normalize(torch.log(pred_map)))
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

def train_w_signal_tracking_diff(
        model:Module,
        optimizer:Optimizer,
        scheduler:LRScheduler,
        loss:Module,
        data:ArrayLike,
        heldout:ArrayLike,
        sig:ArrayLike,
        sig_heldout:ArrayLike,
        diag_start:int,
        diag_end:int,
        dev:device=None,
        num_epoch:int=100,
        early_thresh:float=1e-5
    ) -> Tuple[Module, Module, ArrayLike, ArrayLike]:
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

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.to(dev)
    init_diag = torch.ones((1, data.shape[1])) * data.shape[1]
    init_diag = init_diag.to(dev)
    best_corr = -torch.inf
    arr_corr = []
    best_loss = torch.inf
    arr_loss = []
    model = model.to(dev)
    sig = sig.to(dev)

    train_ii = diagonal_region_indices_from(data, diag_start, diag_end)

    # Turn on interactive mode
    plt.ion()

    # Create the figure and axes objects
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex="col")

    # Initialize lines for updating the plots
    loss_line, = ax1.plot([], [], label='Loss')
    correlation_line, = ax2.plot([], [], color='orange', label='Correlation')

    # Set up the plot parameters
    #ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    #ax1.set_title('Training Loss Over Time')
    ax1.legend()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Correlation')
    #ax2.set_title('Training Correlation Over Time')
    ax2.legend()
    display(fig)

    try:
        for e in range(num_epoch):
            curr_lr = scheduler.get_last_lr()[-1]
            if curr_lr < early_thresh:
                break
            optimizer.zero_grad()
            loss_total = 0
            with torch.no_grad():
                pred_map = model.contact_map_prediction(sig, init_diag)
                pred_map = torch.exp(diagonal_normalize(torch.log(pred_map)))
                pred_map_heldout = model.contact_map_prediction(sig_heldout, init_diag)
                pred_map_heldout = torch.exp(diagonal_normalize(torch.log(pred_map_heldout)))
                curr_cor = mat_corr(pred_map[train_ii] - pred_map_heldout[train_ii],
                                    data[train_ii] - heldout[train_ii])
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

                # Update the data of the plot lines
            if e%100 == 0:
                loss_line.set_data(range(len(arr_loss)), arr_loss)
                correlation_line.set_data(range(len(arr_corr)), arr_corr)

                # Adjust the limits of the plots to fit the data
                ax1.set_xlim(0, len(arr_loss))
                ax1.set_ylim(min(arr_loss), max(arr_loss))

                ax2.set_xlim(0, len(arr_corr))
                ax2.set_ylim(min(arr_corr), max(arr_corr))

                # Redraw the figure to update the display
                fig.canvas.draw()
                fig.canvas.flush_events()
                clear_output(wait=True)
                display(fig)
        print()

    except KeyboardInterrupt:
        print()
        print("Interruption: returning the current best models!")

    finally:
        plt.ioff()
        plt.show()

    return best_loss_model, best_corr_model, np.array(arr_loss), np.array(arr_corr)

def read_json(path:str) -> dict:
    """read the content of a json file and return it as a dictionary.

    Args:
        path (str): path to json file.

    Returns:
        dict: dictionary of the content within the json file.
    """
    with open(path, "r", encoding="utf-8") as file:
        meta = json.loads(file.read())
    return meta

def flip_diag_row(mat:ArrayLike) -> ArrayLike:
    """Swap row and diagonal elements of a given matrix.

    Args:
        mat (ArrayLike): given matrix.

    Returns:
        ArrayLike: row diagonal swapped matrix. 
    """
    n = mat.shape[0]
    ii = np.arange(n)
    iy = ii.reshape(1,-1) * np.ones(n).reshape(-1,1)
    ix = (ii[::-1].reshape(-1,1) - ii[::-1].reshape(1,-1)) % n
    return mat[ix.astype(int), iy.astype(int)]


def diag_indices_in_flipped(mat:ArrayLike,
                            on_diag:int,
                            off_diag:int) -> Tuple[ArrayLike, ArrayLike]:
    """Return the indices for the diagonal in a flipped matrix.

    Args:
        mat (ArrayLike): reference matrix.
        on_diag (int): start diagonal.
        off_diag (int): stop diagonal.

    Returns:
        Tuple[ArrayLike, ArrayLike]: indices for the diagonals.
    """
    return np.where(flip_diag_row(np.tril(np.triu(np.ones_like(mat,dtype=bool),
                                                on_diag),
                                        off_diag-1)
                                )
                )

def diag_indices(mat:ArrayLike,
                 on_diag:int,
                 off_diag:int,
                 tril:bool=False) -> Tuple[ArrayLike, ArrayLike]:
    """Return the indices for the diagonal in a matrix.

    Args:
        mat (ArrayLike): reference matrix.
        on_diag (int): start diagonal.
        off_diag (int): stop diagonal.
        tril (bool): return the indices in lower triangle.

    Returns:
        Tuple[ArrayLike, ArrayLike]: indices for the diagonals.
    """
    if tril:
        return np.where(np.tril(np.triu(np.ones_like(mat,dtype=bool),
                                        on_diag),
                                off_diag-1).T
                       )
    else:
        return np.where(np.tril(np.triu(np.ones_like(mat,dtype=bool),
                                        on_diag),
                                off_diag-1)
                       )

def convert_diags_to_full_contact(diagonals:ArrayLike, start_diag:int, stop_diag:int) -> ArrayLike:
    """Convert diagonals to contact map.

    Args:
        diagonals (ArrayLike): array of the diagonals.
        start_diag (int): start diagonal. 
        stop_diag (int): stop diagonal. 

    Returns:
        ArrayLike: contact matrix.
    """
    patch_size = int(len(diagonals)/(stop_diag-start_diag)+(stop_diag+start_diag-1)/2)
    out = np.full((patch_size, patch_size), np.nan)
    triu_ii = diag_indices_in_flipped(out, start_diag, stop_diag)
    tril_ii = diag_indices(out, start_diag, stop_diag, tril=True)
    out[triu_ii] = diagonals
    out = flip_diag_row(out)
    out[tril_ii] = 0
    return out + out.T

def diag_index_for_mat(size:int, start_diag:int, stop_diag:int) -> callable:
    """Return function that helps with indexing the diagonals in the flatten version

    Args:
        size (int): length of the one dimension of contact map.
        start_diag (int): start diagonal. 
        stop_diag (int): stop diagonal. 

    Returns:
        callable: indexing matrix.
    """
    def indexing(index):
        if index < start_diag:
            raise IndexError(f"Target window for diagonals starts with {start_diag}th diagonal.")
        if index >= stop_diag:
            raise IndexError(f"Target window for diagonals stops at {stop_diag}th diagonal.")
        largest_diagonal = size - start_diag
        offset = index - start_diag
        start_index = largest_diagonal*offset - (offset)*(offset-1)/2
        end_index = start_index + largest_diagonal - offset
        return range(int(start_index), int(end_index))
    return indexing

def read_regions(path:str, res:int, patch_size:int) -> pd.DataFrame:
    """Read regions from the path, provided as a bed file.

    Args:
        path (str): file path.
        res (int): desired resolution, should match contact map's.
        patch_size (int): one dimension of the square contact map patch.

    Returns:
        pd.DataFrame: a DataFrame of regions
    """
    regions = pd.read_csv(path, sep="\t", header=None)
    regions.columns = ["chr", "start", "end", "fold"] 
    regions["start"] = regions["start"]//res * res
    regions["end"] = regions["start"] + patch_size * res
    return regions
