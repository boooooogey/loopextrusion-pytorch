"""Training script for the DLEM model.
"""
import argparse
import os
import torch
import numpy as np
from torch import optim
import dlem
from dlem import util
import lightning as L

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

    def training_step(self, batch, batch_idx):
        """Training step for the model.
        """

        if next(self.model.parameters()).device != self.device_model:
            self.model = self.model.to(self.device_model)

        depth = np.random.choice(range(1, self.depth))
        seq, diagonals, tracks = batch
        batch_size = seq.shape[0]
        out = self.model(diagonals, tracks, seq, depth)
        offset = (2*self.patch_dim - 2*self.start - depth + 1) * depth // 2
        loss = self.loss(out, diagonals[:, offset:])
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for the model.
        """

        if next(self.model.parameters()).device != self.device_model:
            self.model = self.model.to(self.device_model)

        seq, diagonals, tracks, names = batch
        batch_size = seq.shape[0]

        diag_init = torch.from_numpy(np.ones((diagonals[0].shape[0], self.patch_dim - self.start),
                                             dtype=np.float32) * self.patch_dim)

        for diagonal, track, name in zip(diagonals, tracks, names):
            out = self.model.contact_map_prediction(track,
                                                    seq,
                                                    diag_init[:track.shape[0]])
            self.log(
                f"test_corr_{name[0]}", 
                util.vec_corr_batch(diagonal[:, self.index_diagonal(self.start)[-1]+1:].cpu(),out),
                batch_size=batch_size
            )
            self.log(
                f"test_loss_{name[0]}",
                self.loss(out, diagonal[:, self.index_diagonal(self.start)[-1]+1:].cpu()),
                batch_size=batch_size
            )

    def validation_step(self, batch, batch_idx):
        """Validation step for the model.
        """

        if next(self.model.parameters()).device != self.device_model:
            self.model = self.model.to(self.device_model)

        seq, diagonals, tracks, names = batch
        batch_size = seq.shape[0]

        diag_init = torch.from_numpy(np.ones((diagonals[0].shape[0], self.patch_dim - self.start),
                                             dtype=np.float32) * self.patch_dim)

        for diagonal, track, name in zip(diagonals, tracks, names):
            out = self.model.contact_map_prediction(track,
                                                    seq,
                                                    diag_init[:track.shape[0]])
            self.log(
                f"validation_corr_{name[0]}", 
                util.vec_corr_batch(diagonal[:, self.index_diagonal(self.start)[-1]+1:].cpu(),out),
                batch_size=batch_size
            )
            self.log(
                f"validation_loss_{name[0]}",
                self.loss(out, diagonal[:, self.index_diagonal(self.start)[-1]+1:].cpu()),
                batch_size=batch_size
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

        
    

def weighted_mse(pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
    """Calculates the weighted mean squared error. The weights are the exponential of the target.

    Args:
        pred (torch.Tensor): prediction from the model.
        target (torch.Tensor): target.

    Returns:
        torch.Tensor: weighted mean squared error.
    """
    return torch.mean((pred - target)**2*torch.exp(target)
)

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
parser.add_argument('--test-fold', type=str, default='fold4', help='Test fold')
parser.add_argument('--val-fold', type=str, default='fold5', help='Validation fold')
parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=25, help='Patience')
parser.add_argument('--num-epoch', type=int, default=250, help='Number of epochs')
parser.add_argument('--head-type', type=str, default='ForkedHead',)
parser.add_argument('--seq-pooler-type', type=str, default='SequencePoolerAttention',)
parser.add_argument('--resolution', type=int, default=10_000, help='Resolution of the contactmap')
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

args = parser.parse_args()

BATCH_SIZE = args.batch_size
#'../../../loopextrusion_data_creation/.data/training_data_res_1000_patch_size_500'
DATA_FOLDER = args.data_folder
TEST_FOLD = args.test_fold
VAL_FOLD = args.val_fold
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

if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SeqDataset = dlem.dataset_dlem.SeqFeatureDataset if USE_SEQ_FEA else dlem.dataset_dlem.SeqDataset

data_train = dlem.dataset_dlem.CombinedDataset(
    SeqDataset(DATA_FOLDER),
    dlem.dataset_dlem.ContactmapDataset(DATA_FOLDER, RES, select_cell_lines=[TRAIN_CELL_LINE]),
    dlem.dataset_dlem.TrackDataset(DATA_FOLDER, select_cell_lines=[TRAIN_CELL_LINE]))

data_train_sub = torch.utils.data.Subset(data_train,
                                     np.where(
                                         np.logical_and(data_train.data_folds != VAL_FOLD,
                                                        data_train.data_folds != TEST_FOLD))[0])
dataloader_train = torch.utils.data.DataLoader(data_train_sub,
                                               batch_size = BATCH_SIZE,
                                               shuffle=True)

data_val_test = dlem.dataset_dlem.CombinedDataset(
    SeqDataset(DATA_FOLDER),
    dlem.dataset_dlem.ContactmapDataset(DATA_FOLDER, RES),
    dlem.dataset_dlem.TrackDataset(DATA_FOLDER))

data_test = torch.utils.data.Subset(data_val_test,
                                    np.where(data_val_test.data_folds == TEST_FOLD)[0])
data_val = torch.utils.data.Subset(data_val_test,
                                   np.where(data_val_test.data_folds == VAL_FOLD)[0])

dataloader_test = torch.utils.data.DataLoader(data_test, batch_size = BATCH_SIZE, shuffle=False)
dataloader_val = torch.utils.data.DataLoader(data_val, batch_size = BATCH_SIZE, shuffle=False)

seq_pooler = get_seq_pooler(args.seq_pooler_type)(
    LAYER_CHANNEL_NUMBERS,
    LAYER_STRIDES)

model = get_header(args.head_type)(data_val_test.patch_dim,
                                   data_val_test.track_dim,
                                   LAYER_CHANNEL_NUMBERS[-1],
                                   data_val_test.start,
                                   data_val_test.stop,
                                   dlem.util.dlem,
                                   seq_pooler,
                                   channel_per_route=NUMBER_OF_CHANNELS_PER_ROUTE)

model_training = LitTrainer(model,
                            LEARNING_RATE,
                            weighted_mse if LOSS_TYPE == 'weighted_mse' else torch.nn.MSELoss(),
                            data_val_test.patch_dim,
                            data_val_test.start,
                            data_val_test.stop,
                            DEPTH,
                            dev)

checkpoints = [
    L.pytorch.callbacks.ModelCheckpoint(
    monitor=f"validation_loss_{celltype}",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename=f"best_validation_loss_{celltype}"
    ) for celltype in ["H1", "HFF"]
]

checkpoints += [L.pytorch.callbacks.ModelCheckpoint(
    monitor="train_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename="best_train_loss")]

checkpoints += [
    L.pytorch.callbacks.ModelCheckpoint(
    monitor=f"validation_corr_{celltype}",
    mode="max",
    save_top_k=1,
    save_last=True,
    filename=f"best_validation_corr_{celltype}"
    ) for celltype in ["H1", "HFF"]
]

trainer = L.Trainer(accelerator="cpu",
                    devices=1,
                    max_epochs=NUM_EPOCH,
                    log_every_n_steps=100,
                    default_root_dir=SAVE_FOLDER,
                    callbacks=checkpoints
)

trainer.fit(model=model_training,
            train_dataloaders=dataloader_train,
            val_dataloaders=dataloader_val)

#trainer.test(model=model_training, dataloaders=dataloader_test)
