"""Testing DLEM on provided dataset.
"""
import argparse
import os
import torch
import numpy as np
from torch import optim
import dlem
from dlem import util
import lightning as L
from IPython import embed
from tqdm import tqdm

def pairwise_corrcoef(x, y):
    x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True))
    y = (y - y.mean(dim=1, keepdim=True))/(y.std(dim=1, keepdim=True))
    return (x * y).mean(dim=1)

class LitTrainer(L.LightningModule):
    def __init__(self, model,
                 learning_rate,
                 loss,
                 patch_dim,
                 start,
                 stop,
                 depth,
                 device):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss
        self.patch_dim = patch_dim
        self.start = start
        self.stop = stop
        self.depth = depth
        self.index_diagonal = util.diag_index_for_mat(self.patch_dim, self.start, self.stop)
        self.device_model = device
        self.model = self.model.to(self.device_model)

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

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SeqDataset = dlem.dataset_dlem.SeqFeatureDataset if USE_SEQ_FEA else dlem.dataset_dlem.SeqDataset

data = dlem.dataset_dlem.CombinedDataset(
    SeqDataset(DATA_FOLDER),
    dlem.dataset_dlem.ContactmapDataset(DATA_FOLDER, RES),
    dlem.dataset_dlem.TrackDataset(DATA_FOLDER))

dataloader = torch.utils.data.DataLoader(data,
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

model = LitTrainer.load_from_checkpoint(CHECKPOINT,
                                        model=model,
                                        learning_rate=0,
                                        loss=torch.nn.MSELoss,
                                        patch_dim=data.patch_dim,
                                        start=data.start,
                                        stop=data.stop,
                                        depth=15,
                                        device=dev)

model_test = model.model
model_test = model_test.eval()
model_test = model_test.to(dev)

diag_init = torch.from_numpy(np.ones((BATCH_SIZE, data.patch_dim - data.start),
                                     dtype=np.float32) * data.patch_dim)

corrs = {cl:[] for cl in data.cell_line_list}
with torch.no_grad():
    for seq, diagonals, tracks, cls in tqdm(dataloader):
        for diagonal, track, cl in zip(diagonals, tracks, cls):
            cl = cl[0]
            pred = model_test.contact_map_prediction(track, seq, diag_init[:seq.shape[0]])
            #corrs[cl].append(torch.diag(torch.corrcoef(torch.concatenate([pred,
            #                            diagonal[:, data.patch_dim-data.start:]], dim=0))[:pred.shape[0], pred.shape[0]:]).detach().cpu().numpy())
            corrs[cl].append(pairwise_corrcoef(
                pred,
                diagonal[:, data.patch_dim-data.start:]).detach().cpu().numpy())

for cl in corrs:
    corrs[cl] = np.concatenate(corrs[cl])

test_out = data.datasets[0].region_bed.copy()

for cells in corrs:
    test_out[cells] = corrs[cells]

test_out.to_csv(SAVE_FILE, sep='\t', index=False, header=True)
