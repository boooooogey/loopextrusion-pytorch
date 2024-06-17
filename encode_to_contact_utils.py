import numpy as np
import os, sys
from numpy.typing import ArrayLike
from torch import optim
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from models.encodetocontact import DLEM
import torch
from util import train_w_signal, diagonal_normalize, plot_results, diagonal_region_indices_from, mat_corr, train_w_signal_tracking_diff
from matplotlib.patches import Patch
import colorcet as cc

def zscore(mat:ArrayLike) -> ArrayLike:
    """Zscore collumns of the matrix.

    Args:
        mat (ArrayLike): input matrix. 

    Returns:
        ArrayLike: column zscored.
    """
    return (mat - mat.mean(axis=-2)[np.newaxis])/(mat.std(axis=-2)[np.newaxis])

def plot_map_diff(map1, map2, signal1d1, signal1d2, label1, label2, color_lookup, axes=None):
    if axes is None:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8.5),
                               sharex='col',
                               sharey='row',
                               gridspec_kw={'wspace':0,
                                           'hspace':0,
                                           'height_ratios':[1,2]})
    for i, map in enumerate([map1, map2, np.abs(map1 - map2)]):
        axes[1,i].matshow(map, cmap="icefire")
        axes[1,i].xaxis.set_ticks_position('bottom')
        axes[1,i].xaxis.set_label_position('bottom')
    for i, (s, l) in enumerate(zip([signal1d1, signal1d2, np.hstack([signal1d1, signal1d2])],
                              [label1, label2, np.concatenate([label1, label2])])):
        for c in range(s.shape[1]):
            axes[0,i].plot(s[:,c], color=color_lookup[l[c]], label=l[c])

    legend_elements = [Patch(facecolor=color, edgecolor='black', label=label) for label, color in color_lookup.items()]

    fig.legend(handles=legend_elements, loc='lower left', fontsize='large')#, title='Labels Legend')

def return_model_result(model, encode):
    with torch.no_grad():
        init = torch.ones((1, encode.shape[1])) * encode.shape[1]
        pred = model.cpu().contact_map_prediction(encode,
                                                init).detach().cpu()
        pred = diagonal_normalize(torch.log(pred))[0]

        params = model.cpu().converter(encode.transpose(-2,-1)).cpu().detach().numpy()
        p_l = params[0,0,:]
        p_r = params[0,1,:]
    return pred, p_l, p_r

def plot_results_comp_train_val(model, cmap_train, cmap_val, encode_train, encode_val, diag_stop):
    pred_train, p_l_train, p_r_train = return_model_result(model, encode_train[None])
    pred_val, p_l_val, p_r_val = return_model_result(model, encode_val[None])

    _, axes = plt.subplots(nrows=3, ncols=6, figsize=(12 * 3,14),
                            sharex='col',
                            sharey='row',
                            gridspec_kw={'wspace':0,
                                        'hspace':0,
                                        'height_ratios':[10, 50, 10],
                                        'width_ratios':[50, 10]*3})

    plot_results(cmap_train,
                 pred_train,
                 (p_l_train, p_r_train, np.ones_like(p_l_train)),
                 ignore_i_off=diag_stop, axes=axes[:,:2])
    plot_results(cmap_val,
                 pred_val,
                 (p_l_val, p_r_val, np.ones_like(p_l_val)),
                 ignore_i_off=diag_stop, axes=axes[:,2:4])
    plot_results(np.abs(cmap_val - cmap_train),
                 np.abs(pred_val - pred_train),
                 (p_l_val, p_r_val, np.ones_like(p_l_val)),
                 ignore_i_off=diag_stop, axes=axes[:,4:])
    for i in range(6):
        axes[2, i].set_visible(False)
    axes[0, 4].set_visible(False)
    axes[1, 5].set_visible(False)
    plt.tight_layout()

def plot_results_comp_params_sig(model, encode_train, encode_val, axes=None):
    if axes is None:
        _, axes = plt.subplots(nrows=encode_train.shape[0]+2, figsize=(10,6))
    _, p_l_train, p_r_train = return_model_result(model, encode_train[None])
    _, p_l_val, p_r_val = return_model_result(model, encode_val[None])
    n = encode_train.shape[1]
    for i in range(encode_train.shape[1]):
        axes[i].plot(encode_train[:,i])
        axes[i].plot(encode_val[:,i])
    axes[n].plot(p_l_train)
    axes[n].plot(p_l_val)
    axes[n+1].plot(p_r_train)
    axes[n+1].plot(p_r_val)

def plot_metrics(loss, corr):
    _, axes = plt.subplots(nrows=2, figsize=(10,6))
    axes[0].plot(loss)
    axes[1].plot(corr)

def sub_select_bed(bed:pd.DataFrame, chr:str, start:int, stop:int) -> ArrayLike:
    """Select part of the bed file. Return it as a numpy array.

    Args:
        bed (pd.DataFrame): bed content as dataframe.
        chr (str): chromosome 
        start (int): start location.
        stop (int): stop location.

    Returns:
        ArrayLike: selected region as numpy array.
    """
    return bed.iloc[np.logical_and(np.logical_and(bed.iloc[:,0] == chr,
                                                  bed.iloc[:,1] >= start),
                                                  bed.iloc[:,2] <= stop).to_numpy()]
 
def return_vec_from_bed(bed:pd.DataFrame, n:int) -> ArrayLike:
    """Return region in a bed file as an expanded vector.

    Args:
        bed (pd.DataFrame): bed file of genomic regions with scores.
        n (int): length of the vector

    Returns:
        ArrayLike: expanded vector.
    """
    out = np.zeros(n)
    for ro in bed.iloc[:,1:].to_numpy(dtype=int):
        out[ro[0]:(ro[1]+1)] = np.maximum(out[ro[0]:(ro[1]+1)], np.ones((ro[1]+1) - ro[0])*ro[2])
    return out

def read_encode(path:str,
                mode:str,
                chr:str,
                start:int,
                stop:int) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Read encode data for two cell lines H1 and HFF.

    Args:
        path (str): folder path 
        mode (str): Mode. Max/Mean. 
        chr (str): chromosome
        start (int): start location.
        stop (int): stop location. 

    Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike]: Tuple of encode signals for two cell lines, and the
        signal labels.
    """
    encode = pd.read_csv(path, sep="\t")
    mode_select = np.where([mode.lower() in i.lower() for i in encode.columns])[0].tolist()
    encode = encode.iloc[:, [1, 2, 3] + mode_select]
    encode = sub_select_bed(encode, chr, start, stop).iloc[:, 3:]
    labels = np.array([label.split("_")[2] for label in encode.columns])
    h1_ii = np.array([n.split("_")[1] == "H1" for n in encode])
    encode = encode.to_numpy(dtype=float)
    return encode[:, h1_ii], encode[:, np.logical_not(h1_ii)], np.unique(labels)

def read_contact_map(map_path:str,
                     start:int,
                     stop:int) -> ArrayLike:
    """Read contact maps for a cell line.

    Args:
        map_path (str): Path to the file.
        start (int): start location
        stop (int): stop location

    Returns:
        ArrayLike: contact  map for the region specified by start stop.
    """
    contact_map = np.load(map_path)[:-1, :-1]
    contact_map[np.isnan(contact_map)] = 0
    return contact_map[start:stop, start:stop]

def read_sequence_features(path:str,
                           n:int,
                           chromosome:str,
                           start:int,
                           stop:int,
                           resolution:int) -> ArrayLike:
    """Read sequence features from a bed file and expand them to a vector.

    Args:
        path (str): bed file path.
        n (int): size of the vector
        chromosome (str): chromosome
        start (int): start location
        stop (int): stop location
        resolution(int): resolution of the bins

    Returns:
        ArrayLike: expanded signal vector.
    """
    ctcf = pd.read_csv(path, sep="\t", header=None)
    ctcf = sub_select_bed(ctcf, chromosome, start, stop)
    ctcf.iloc[:,1] = (ctcf.iloc[:,1]-start) // resolution
    ctcf.iloc[:,2] = (ctcf.iloc[:,2]-start) // resolution 
    ctcf_vec = return_vec_from_bed(ctcf, n)
    return ctcf_vec

def read_data(h1_map_path:str, hff_map_path:str, encode_path:str, seq_fea_path_pos:str,
              seq_fea_path_neg:str, mode:str, chromosome:str, start:int, stop:int,
              res:int) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Read the data for the experiment.

    Args:
        h1_map_path (str): _description_
        hff_map_path (str): _description_
        encode_path (str): _description_
        seq_fea_path_pos (str): _description_
        seq_fea_path_neg (str): _description_
        mode (str): _description_
        chromosome (str): _description_
        start (int): _description_
        stop (int): _description_
        res (int): _description_

    Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]: _description_
    """
    stop_binned = stop//res
    start_binned = start//res
    n = stop_binned - start_binned 
    map_h1 = read_contact_map(h1_map_path, start_binned, stop_binned)
    map_hff = read_contact_map(hff_map_path, start_binned, stop_binned)
    encode_h1, encode_hff, signal_names = read_encode(encode_path, mode, chromosome, start, stop)
    ctcf_neg = read_sequence_features(seq_fea_path_pos, n, chromosome, start, stop, res)
    ctcf_pos = read_sequence_features(seq_fea_path_neg, n, chromosome, start, stop, res)
    encode_h1 = np.hstack([encode_h1,
                           ctcf_neg[:,np.newaxis], ctcf_pos[:,np.newaxis]])
    encode_hff = np.hstack([encode_hff,
                            ctcf_neg[:,np.newaxis], ctcf_pos[:,np.newaxis]])
    return (diagonal_normalize(map_h1[np.newaxis])[0],
            diagonal_normalize(map_hff[np.newaxis])[0],
            zscore(encode_h1),
            zscore(encode_hff),
            signal_names)

def assign_unique_colors(labels:ArrayLike) -> dict:
    """Assign unique colors to signal names.

    Args:
        signal_names (ArrayLike): names of the signals.

    Returns:
        dict: mapping of colors and labels.
    """
    color_n = len(labels)
    palette = seaborn.color_palette(cc.glasbey, n_colors=color_n)
    return {sig:palette[n] for n, sig in enumerate(labels)}

def run_training(map_train,
                 map_val,
                 encode_train,
                 encode_val,
                 dev,
                 learning_rate=0.001,
                 patience=20,
                 diag_start=3,
                 diag_stop=50,
                 num_epoch=1500,
                 model=None):
    if model is None:
        model = DLEM(map_train.shape[1], encode_train.shape[2])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, mode="max")

    return train_w_signal_tracking_diff(
        model,
        optimizer,
        scheduler,
        torch.nn.MSELoss(reduction="sum"),
        torch.exp(map_train).to(dev),
        torch.exp(map_val).to(dev),
        encode_train.to(dev),
        encode_val.to(dev),
        diag_start,
        diag_stop,
        dev,
        num_epoch=num_epoch
    )

def run_experiment(mode, chromosome, start, stop, resolution, folder):

    map_h1, map_hff, encode_h1, encode_hff, signal_names = read_data(
                                    ".data/ForGrant/chr10_8k_H1.npy",
                                    ".data/ForGrant/chr10_8k_HFF.npy",
                                    ".data/ForGrant/bestTracksBinned.txt",
                                    "../loopExtraction/data/ctcf/MA0139.1.neg.score.sorted.bedgraph",
                                    "../loopExtraction/data/ctcf/MA0139.1.pos.score.sorted.bedgraph",
                                    mode,
                                    chromosome,
                                    start,
                                    stop,
                                    resolution)

    signal_names = np.concatenate([signal_names, ['CTCF_neg', 'CTCF_pos']])
    color_dict = assign_unique_colors(signal_names)

    plot_map_diff(map_h1, map_hff, encode_h1, encode_hff, signal_names, signal_names, color_dict)
    plt.savefig(os.path.join(folder, "data.png"))
    plt.close()

    dev = torch.device('cuda')

    map_h1 = torch.tensor(map_h1, dtype=torch.float32)[None]
    map_hff = torch.tensor(map_hff, dtype=torch.float32)[None]
    encode_h1 = torch.tensor(encode_h1, dtype=torch.float32)[None]
    encode_hff = torch.tensor(encode_hff, dtype=torch.float32)[None]

    best_loss_model, best_corr_model, arr_loss, arr_corr = run_training(map_h1,
                                                                        map_hff,
                                                                        encode_h1,
                                                                        encode_hff,
                                                                        dev,
                                                                        learning_rate=0.001,
                                                                        patience=100000,
                                                                        diag_start=3,
                                                                        diag_stop=DIAG_STOP,
                                                                        num_epoch=1500)

    plot_results_comp_train_val(best_corr_model, map_h1[0],
                                map_hff[0], encode_h1[0], encode_hff[0],
                                DIAG_STOP)

    plt.savefig(os.path.join(folder, "results.png"))
    plt.close()

    plot_metrics(arr_loss, arr_corr)
    plt.savefig(os.path.join(folder, "metrics.png"))
    plt.close()
