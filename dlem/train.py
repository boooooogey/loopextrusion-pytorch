"""Training script for the DLEM model.
"""
import argparse
import os
import torch
import numpy as np
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
import dlem
from dlem.trainer import LitTrainer
from dlem.trainer_data import LitTrainerData

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

NUMBER_OF_CHANNELS_PER_ROUTE = 3

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--data-folder', type=str, required=True, help='Data folder')
parser.add_argument('--save-folder', type=str, required=True, help='Save folder')
parser.add_argument('--batch-size', type=int, default=5, help='Batch size')
parser.add_argument('--offset-contactmaps', type=int, nargs='+', required=True,
                    help='Offsets for the contactmaps')
parser.add_argument('--overlap-contactmaps', type=int, required=True,
                    help='Overlap for the contactmaps')
parser.add_argument('--patch-size', type=int, required=True,
                    help='Patchsize for the contactmaps')
parser.add_argument('--project-name', type=str, default="DLEM_runs", help='Project name for wandb.')
parser.add_argument('--test-chrom', type=str, default='chr9', help='Test fold')
parser.add_argument('--val-chrom', type=str, default='chr8', help='Validation fold')
parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--head-layer-num', type=int, default=4, help='Head layer number')
parser.add_argument('--patience', type=int, default=25, help='Patience')
parser.add_argument('--num-epoch', type=int, default=250, help='Number of epochs')
parser.add_argument('--head-type', type=str, default='ForkedHead',)
parser.add_argument('--seq-pooler-type', type=str, default='SequencePoolerAttention',)
parser.add_argument('--seq-dim', type=int, default=2,)
parser.add_argument('--resolution', type=int, default=10_000, help='Resolution of the contactmap')
parser.add_argument('--save-file', type=str, default=None, help='Save the correlations as a file')
parser.add_argument('--layer-channel-numbers', type=int, nargs='+', default=[4,8,8,8,8],
                    help='Channel numbers for convolutional layers')
parser.add_argument('--layer-strides', type=int, nargs='+', default=[10,10,10,10],
                    help='Strides for convolutional layers')
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'weighted_mse'],
                    help='Loss function')
parser.add_argument('--depth', type=int, default=3,
                    help='How further the model should be trained on')
parser.add_argument('--training-cell-line', type=str, default="H1",
                    help="Select on which cell line the model will be trained on")
parser.add_argument('--use-seq-feat', action='store_true',
                    help='Use sequence features instead of sequence')
parser.add_argument('--number-of-channel-per-route', type=int, default=3)
parser.add_argument('--pool-bigwigs', action='store_true', help='Pool bigwigs')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
#'../../../loopextrusion_data_creation/.data/training_data_res_1000_patch_size_500'
DATA_FOLDER = args.data_folder
TEST_CHROM = args.test_chrom
VAL_CHROM = args.val_chrom
OVERLAP = args.overlap_contactmaps
OFFSETS = args.offset_contactmaps
PATCH_SIZE = args.patch_size
LEARNING_RATE = args.learning_rate
PATIENCE = args.patience
NUM_EPOCH = args.num_epoch
SAVE_FOLDER = args.save_folder
#'/data/genomes/human/Homo_sapiens/UCSC/hg38/Sequence/WholeGenomeFasta/genome.fa'
LOSS_TYPE = args.loss
LR_THRESHOLD = 1.5e-7
LAYER_CHANNEL_NUMBERS = args.layer_channel_numbers
LAYER_STRIDES = args.layer_strides
RES = args.resolution
DEPTH = args.depth
TRAIN_CELL_LINE = args.training_cell_line
USE_SEQ_FEA = args.use_seq_feat
NUMBER_OF_CHANNELS_PER_ROUTE = args.number_of_channel_per_route
SAVE_FILE = args.save_file
HEAD_LAYER_NUM = args.head_layer_num
POOL_BIGWIGS = args.pool_bigwigs
SEQ_DIM = args.seq_dim
PROJECT_NAME = args.project_name

if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)

#dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_train = dlem.dataset_dlem.DlemData(
    DATA_FOLDER,
    RES,
    PATCH_SIZE,
    subselection=[TRAIN_CELL_LINE],
    overlap=OVERLAP,
    offset=OFFSETS[0],
    chrom_filter=[TEST_CHROM, VAL_CHROM],
    #chrom_selection=['chr7'],
    pool_bigwigs=POOL_BIGWIGS
    )

data_val = dlem.dataset_dlem.DlemData(
    DATA_FOLDER,
    RES,
    PATCH_SIZE,
    overlap=OVERLAP,
    offset=0,
    chrom_selection=[VAL_CHROM],
    pool_bigwigs=POOL_BIGWIGS
    )

data_test = dlem.dataset_dlem.DlemData(
    DATA_FOLDER,
    RES,
    PATCH_SIZE,
    overlap=OVERLAP,
    offset=0,
    chrom_filter=[TEST_CHROM],
    pool_bigwigs=POOL_BIGWIGS
    )

trainer_data = LitTrainerData(data_train, data_test, data_val, BATCH_SIZE, OVERLAP, OFFSETS)

seq_pooler = get_seq_pooler(args.seq_pooler_type)(
    LAYER_CHANNEL_NUMBERS,
    LAYER_STRIDES)

model = get_header(args.head_type)(data_train.patch_size,
                                   data_train.track_dim,
                                   SEQ_DIM, #LAYER_CHANNEL_NUMBERS[-1],
                                   data_train.start,
                                   data_train.stop,
                                   dlem.util.dlem,
                                   seq_pooler,
                                   channel_per_route=NUMBER_OF_CHANNELS_PER_ROUTE,
                                   layer_num=HEAD_LAYER_NUM)

model_training = LitTrainer(model,
                            LEARNING_RATE,
                            weighted_mse if LOSS_TYPE == 'weighted_mse' else torch.nn.MSELoss(),
                            data_train.patch_size,
                            data_train.start,
                            data_train.stop,
                            DEPTH,
                            metric_file_path=SAVE_FILE)

checkpoints = [
    L.pytorch.callbacks.ModelCheckpoint(
    monitor=f"validation_loss_{celltype}",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename=f"best_validation_loss_{celltype}",
    dirpath=SAVE_FOLDER
    ) for celltype in ["H1", "HFF"]
]

checkpoints += [L.pytorch.callbacks.ModelCheckpoint(
    monitor="train_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename="best_train_loss",
    dirpath=SAVE_FOLDER
)]

checkpoints += [
    L.pytorch.callbacks.ModelCheckpoint(
    monitor=f"validation_corr_{celltype}",
    mode="max",
    save_top_k=1,
    save_last=True,
    filename=f"best_validation_corr_{celltype}",
    dirpath=SAVE_FOLDER
    ) for celltype in ["H1", "HFF"]
]

checkpoints += [
    L.pytorch.callbacks.ModelCheckpoint(
    monitor="validation_corr_diff",
    mode="max",
    save_top_k=1,
    save_last=True,
    filename="best_validation_corr_diff",
    dirpath=SAVE_FOLDER)
]

wandb.login(key="d4cd96eb50ccb5168c4b750d269715d2cfbd8e44")
wandb_logger = WandbLogger(name=f"cell_line_{TRAIN_CELL_LINE}_channel_per_route_{NUMBER_OF_CHANNELS_PER_ROUTE}_seq_pooler_{args.seq_pooler_type}_head_{args.head_type}_loss_{LOSS_TYPE}_lr_{LEARNING_RATE}_depth_{DEPTH}",
                           project=PROJECT_NAME,
                           save_dir=SAVE_FOLDER)

trainer = L.Trainer(accelerator='cuda',
                    devices=1,
                    max_epochs=NUM_EPOCH,
                    default_root_dir=SAVE_FOLDER,
                    callbacks=checkpoints,
                    logger=wandb_logger
)

trainer.fit(model=model_training,
            datamodule=trainer_data)

trainer.test(model_training, datamodule=trainer_data)
