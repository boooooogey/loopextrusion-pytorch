from typing import List, Union, Any
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from dlem.loader import load_model, load_reader
from dlem.feature_extraction import extractor
from dlem.util import diagonal_normalize
from dlem.util import get_detachment_rate


def initiate_dataframe(column_names:List[str], length:int) -> pd.DataFrame:
    """_summary_

    Args:
        column_names (_type_): _description_
        length (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Define the column names and their data types
    column_types = [str, int, int, int] + [float] * (len(column_names) - 4)
    column_init = dict(zip(column_names, column_types))

    # Preallocate the data
    data = {col: np.empty(length, dtype=dtype) for col, dtype in column_init.items()}

    # Create the DataFrame
    return pd.DataFrame(data)


'''
def weighted_mse(output:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
    """
    Calculates the weighted mean squared error loss between the output and target tensors.

    Args:
        output (torch.Tensor): The predicted output tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The calculated weighted mean squared error loss.
    """
    loss = torch.mean((output - target)**2 * torch.exp(target))
    return loss
'''

# replace this with your functions

def weighted_mse(input, target, weight=None):
    if weight is None:
        weight = torch.exp(target)
    return torch.mean(weight * (input - target) ** 2)

def wMSE_weight(weight_matrix):
    return weight_matrix


# here, add another optimization where you find the detach!




def extract_param_from_mcool(cooler_file:str,
                             output_path:str,
                             model_name:str,
                             window_size:int,
                             stride:int,
                             resolution:int,
                             chrom_subset:Union[List[str],None]=None,
                             do_fit_detach:bool=False,
                             perc_nan_threshold:float=0.3,
                             lr:float=0.5,
                             reader_name:str="datareader_cooler",
                             dev_name:str='cuda',
                             do_return_result:bool=False,
                             **kwargs:Any):

    reader_arch = load_reader(reader_name)

    # Initiate a dummy model to get the parameter names
    model_tmp = load_model(model_name)(1,10000)

    data = reader_arch(cooler_file, resolution, window_size, stride, chrom_subset=chrom_subset)

    sample_n = len(data) * window_size

    # Initiate Pandas DataFrame to store the results
    column_names = ['chrom', 'start', 'end', 'patch_i', 'max_corr', 'perc_nan', 'importance_weights']      # add weights for averaging: product of distance + coverage
    column_names += model_tmp.return_parameter_names()
    results = initiate_dataframe(column_names, sample_n)

    diag_stop = kwargs.get('diag_stop', 1_200_000 // resolution)
    diag_start = kwargs.get('diag_start',  50_000 // resolution)

    depth = 10 * int(10_000/resolution)

    weights = wMSE_weight 

    distance = np.minimum(np.arange(window_size), window_size - 1 - np.arange(window_size)) + 1
    # Apply upper bound
    distance = np.minimum(distance, diag_stop)


    #( cooler_file: str, resolution: int, chromosome: str, start: int, end: int, do_adaptive_coarsegrain: bool = True, return_raw: bool = False)
    if do_fit_detach:
        detachment = 0
        for chrID in chrom_subset:
            detachment += get_detachment_rate(resolution, dlem.loader.get_contact_map(cooler_file,
                                                                                resolution,
                                                                                chr=chrID,
                                                                                start=0,
                                                                                end=100, # whatever the end of that chr is; must sort it.
                                                                                do_adaptive_coarsegrain = False)[0]
                                                                                )
        detachment /= len(chrom_subset)   
    # this is the first version. the second version could be chr-specific. 
    
    else:
        res_detach = {10000:0.025,
                      5000:0.0125,
                      2000:0.005}
        detachment = [res_detach[resolution]]

    for i, (patch, perc_nan, _, _, _) in tqdm(enumerate(data), total=len(data)):
        try:
            if perc_nan > perc_nan_threshold:
                out = ([np.nan] * len(model_tmp.return_parameter_names()), np.nan)
            else:
                out = extractor(patch=diagonal_normalize(np.log(patch)[np.newaxis])[0],
                                res=resolution,
                                detach=detachment,
                                learning_rate=lr,
                                arch=model_name,
                                diag_start=diag_start,
                                diag_stop=diag_stop,
                                depth=depth,
                                loss=weighted_mse,
                                weights=weights(patch),
                                dev_name=dev_name,
                                **kwargs)
            
            chr_arr, start_arr, end_arr, pd_indx = data.return_chrom_positions(i)

            results.iloc[pd_indx, 0] = chr_arr
            results.iloc[pd_indx, 1] = start_arr
            results.iloc[pd_indx, 2] = end_arr
            results.iloc[pd_indx, 3] = i
            results.iloc[pd_indx, 4] = out[-1]
            results.iloc[pd_indx, 5] = perc_nan
            results.iloc[pd_indx, 6] = distance * (1-perc_nan)
            for p_i, param in enumerate(out[0]):
                results.iloc[pd_indx, 7 + p_i] = param
        except KeyboardInterrupt:
            break
        if i == len(data)-1:
            break

    # Write results dataframe to a given path as a tsv
    results.to_csv(output_path, sep='\t', index=False)

    if do_return_result:
        return results





def parse_arguments():
    """
    Parse the terminal arguments for the dlem function.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='DLEM')
    parser.add_argument('cooler_file', type=str, help='Path to the cooler file')
    parser.add_argument('output_path', type=str, help='Path to save the output')
    parser.add_argument('resolution', type=int, help='Resolution')
    parser.add_argument('--stride', type=int, help='Stride')
    parser.add_argument('--window-size', type=int, help='Window size')
    parser.add_argument('--model-name', type=str, default="netdlem2", help='Name of the model')
    parser.add_argument('--chrom-subset', nargs='+', help='Subset of chromosomes')
    parser.add_argument('--perc-nan-threshold', type=float, default=0.3,
                        help='Percentage NaN threshold')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--reader-name', type=str, default='datareader_cooler',
                        help='Name of the reader')

    args = parser.parse_args()

    if args.window_size is None:
        args.window_size = 2_000_000 // args.resolution

    if args.stride is None:
        args.stride = args.window_size // 4

    return args

def dlem():
    """DLEM user interface.
    """
    args = parse_arguments()
    extract_param_from_mcool(args.cooler_file,
                             args.output_path,
                             args.model_name,
                             args.window_size,
                             args.stride,
                             args.resolution,
                             chrom_subset=args.chrom_subset,
                             perc_nan_threshold=args.perc_nan_threshold,
                             lr=args.lr,
                             reader_name=args.reader_name)


def dlem_notebook(cooler_file, output_path, resolution, 
                 model_name="minimal_dlem", window_size=None, stride=None,
                 chrom_subset=None, perc_nan_threshold=0.3, lr=0.5, 
                 reader_name='datareader_cooler', dev_name='cuda', do_return_result:bool=False):
    """DLEM interface for Jupyter notebooks.
    
    Parameters
    ----------
    cooler_file : str
        Path to the .mcool file
    output_path : str
        Directory to save outputs
    resolution : int
        Resolution of the contact matrix
    model_name : str, optional
        Name of the model to use (default: "netdlem2")
    window_size : int, optional
        Window size for processing (default: 2_000_000 // resolution)
    stride : int, optional
        Stride length for processing (default: window_size // 4)
    chrom_subset : list, optional
        List of chromosomes to process
    perc_nan_threshold : float, optional
        Threshold for NaN percentage (default: 0.3)
    lr : float, optional
        Learning rate (default: 0.5)
    reader_name : str, optional
        Name of the reader (default: 'datareader_cooler')
    """
    # Calculate default values for window_size and stride if not provided
    if window_size is None:
        window_size = 2_000_000 // resolution
    if stride is None:
        stride = window_size // 4
        
    # Call the original function with the provided parameters
    
    df_results = extract_param_from_mcool(cooler_file,
                           output_path,
                           model_name,
                           window_size,
                           stride,
                           resolution,
                           chrom_subset=chrom_subset,
                           perc_nan_threshold=perc_nan_threshold,
                           lr=lr,
                           reader_name=reader_name,
                           dev_name=dev_name, do_return_result=do_return_result)
    
    if do_return_result:
        return df_results
