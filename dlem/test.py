"""Testing DLEM on provided dataset.
"""
import argparse
import numpy as np
import torch
import lightning as L
import dlem
from dlem.trainer import LitTrainer

def weighted_mse(pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
    """Calculates the weighted mean squared error. The weights are the exponential of the target.

    Args:
        pred (torch.Tensor): prediction from the model.
        target (torch.Tensor): target.

    Returns:
        torch.Tensor: weighted mean squared error.
    """
    loss = torch.mean((pred - target)**2*torch.exp(target))
    if torch.any(torch.isnan(loss)):
        np.save(".local/debug/weighted_mse_pred.npy", pred.detach().cpu().numpy())
        np.save(".local/debug/weighted_mse_target.npy", target.detach().cpu().numpy())
    return loss

def get_seq_pooler(class_name:str) -> dlem.seq_pooler.SequencePooler:
    """Imports a sequence pooler class from the seq_pooler module.

    Args:
        class_name (str): name of the sequence pooler.

    Returns:
        dlem.seq_pooler.SequencePooler: sequence pooler class
    """

    module = __import__('dlem.seq_pooler', fromlist=[class_name])
    seqclass = getattr(module, class_name)
    return seqclass

def get_header(class_name:str) -> dlem.head.BaseHead:
    """Imports a head class from the head module.

    Args:
        class_name (str): name of the head class.

    Returns:
        dlem.head.Head: head class
    """
    module = __import__('dlem.head', fromlist=[class_name])
    headclass = getattr(module, class_name)
    return headclass


parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--data-folder', type=str, required=True, help='Data folder')
parser.add_argument('--model-checkpoint', type=str, required=True,
                    help='Path to the model checkpoint')
parser.add_argument('--save-file', type=str, required=True, help='Save the correlations as a file')
parser.add_argument('--batch-size', type=int, default=5, help='Batch size')
parser.add_argument('--head-type', type=str, default='ForkedHead',)
parser.add_argument('--seq-pooler-type', type=str, default='SequencePoolerAttention',)
parser.add_argument('--resolution', type=int, default=10_000, help='Resolution of the contactmap')
parser.add_argument('--layer-channel-numbers', type=int, nargs='+', default=[4,8,8,8,8],
                    help='Channel numbers for convolutional layers')
parser.add_argument('--layer-strides', type=int, nargs='+', default=[10,10,10,10],
                    help='Strides for convolutional layers')
parser.add_argument('--use-seq-feat', action='store_true',
                    help='Use sequence features instead of sequence')
parser.add_argument('--number-of-channel-per-route', type=int, default=3)
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'weighted_mse'],
                    help='Loss function')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
#'../../../loopextrusion_data_creation/.data/training_data_res_1000_patch_size_500'
DATA_FOLDER = args.data_folder
CHECKPOINT = args.model_checkpoint
SAVE_FILE = args.save_file
#'/data/genomes/human/Homo_sapiens/UCSC/hg38/Sequence/WholeGenomeFasta/genome.fa'
LAYER_CHANNEL_NUMBERS = args.layer_channel_numbers
LAYER_STRIDES = args.layer_strides
RES = args.resolution
USE_SEQ_FEA = args.use_seq_feat
NUMBER_OF_CHANNELS_PER_ROUTE = args.number_of_channel_per_route
LOSS_TYPE = args.loss

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SeqDataset = dlem.dataset_dlem.SeqFeatureDataset if USE_SEQ_FEA else dlem.dataset_dlem.SeqDataset

data = dlem.dataset_dlem.CombinedDataset(
    SeqDataset(DATA_FOLDER),
    dlem.dataset_dlem.ContactmapDataset(DATA_FOLDER, RES),
    dlem.dataset_dlem.TrackDataset(DATA_FOLDER))

#data_val = torch.utils.data.Subset(data,
#                                   np.where(data.data_folds == "fold5")[0])

dataloader = torch.utils.data.DataLoader(data, #data_val,
                                         batch_size = BATCH_SIZE,
                                         shuffle=False)

seq_pooler = get_seq_pooler(args.seq_pooler_type)(
    LAYER_CHANNEL_NUMBERS,
    LAYER_STRIDES)

model = get_header(args.head_type)(data.patch_dim,
                                   data.track_dim,
                                   LAYER_CHANNEL_NUMBERS[-1],
                                   data.start,
                                   data.stop,
                                   dlem.util.dlem,
                                   seq_pooler,
                                   channel_per_route=NUMBER_OF_CHANNELS_PER_ROUTE)

#model = LitTrainer.load_from_checkpoint(checkpoint_path=CHECKPOINT,
#                                        model=model,
#                                        learning_rate=0,
#                                        loss=torch.nn.MSELoss(),
#                                        patch_dim=data.patch_dim,
#                                        start=data.start,
#                                        stop=data.stop,
#                                        depth=15,
#                                        device=dev,
#                                        metric_file_path=SAVE_FILE)
model = LitTrainer.load_from_checkpoint(checkpoint_path=CHECKPOINT,
                                        model=model,
                                        metric_file_path=SAVE_FILE,
                                        loss=weighted_mse if LOSS_TYPE == 'weighted_mse' else torch.nn.MSELoss())

trainer = L.Trainer(accelerator="cpu", devices=1)

trainer.test(model, dataloader)
