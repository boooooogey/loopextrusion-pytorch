"""Training script for the DLEM model.
"""
import argparse
import time
import os
import torch
import numpy as np
from torch import optim
import dlem
from dlem import util

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

if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = dlem.dataset_dlem.CombinedDataset(dlem.dataset_dlem.SeqDataset(DATA_FOLDER),
                                         dlem.dataset_dlem.ContactmapDataset(DATA_FOLDER, RES),
                                         dlem.dataset_dlem.TrackDataset(DATA_FOLDER))

data_test = torch.utils.data.Subset(data, np.where(data.data_folds == TEST_FOLD)[0])
data_val = torch.utils.data.Subset(data, np.where(data.data_folds == VAL_FOLD)[0])
data_train = torch.utils.data.Subset(data, np.where(np.logical_and(data.data_folds != VAL_FOLD,
                                                                data.data_folds != TEST_FOLD))[0])

dataloader_test = torch.utils.data.DataLoader(data_test, batch_size = BATCH_SIZE, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(data_val, batch_size = BATCH_SIZE, shuffle=True)
dataloader_train = torch.utils.data.DataLoader(data_train, batch_size = BATCH_SIZE, shuffle=True)

index_diagonal = util.diag_index_for_mat(data.patch_dim, data.start, data.stop)

seq_pooler = get_seq_pooler(args.seq_pooler_type)(
    LAYER_CHANNEL_NUMBERS,
    LAYER_STRIDES)

model = get_header(args.head_type)(data.patch_dim,
                                   data.track_dim,
                                   LAYER_CHANNEL_NUMBERS[-1],
                                   data.start,
                                   data.stop,
                                   dlem.util.dlem,
                                   seq_pooler)

print(model)
model = model.to(dev)

if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE, mode='max')

assert LOSS_TYPE in ["mse", "weighted_mse"]
if LOSS_TYPE == "mse":
    loss = torch.nn.MSELoss(reduction='mean')
else:
    loss = weighted_mse

diag_init = torch.from_numpy(np.ones((BATCH_SIZE, data.patch_dim - data.start),
                                     dtype=np.float32) * data.patch_dim)


best_loss = torch.inf
best_val_loss = torch.inf
best_corr = -torch.inf
mean_loss_traj_train = []
mean_corr_traj_val = []
mean_loss_traj_val = []
read_times = []
inner_times = []
model = model.to(dev)
diag_init = diag_init.to(dev)
for e in range(NUM_EPOCH):
    training_loss = []
    validation_corr = []
    validation_loss = []
    model.train()
    end = time.time()
    for seq, diagonals, tracks in dataloader_train:
        depth = np.random.choice(range(1, DEPTH))
        start = time.time()
        read_times.append(start-end)
        optimizer.zero_grad()
        out = model(diagonals, tracks, seq, depth)
        offset = (2*data.patch_dim - 2*data.start - depth + 1) * depth // 2
        total_loss = loss(out, diagonals[:, offset:])
        total_loss.backward()
        optimizer.step()
        training_loss.append(total_loss.detach().cpu().numpy())
        end = time.time()
        inner_times.append(end-start)

    mean_total_loss = np.mean(training_loss)
    mean_loss_traj_train.append(mean_total_loss)

    if mean_total_loss < best_loss:
        best_loss = mean_total_loss
        torch.save(model.state_dict(),
                   os.path.join(SAVE_FOLDER, "best_loss.pt"))

    with torch.no_grad():
        model.eval()

        #end = time.time()
        for seq, diagonals, tracks in dataloader_val:
            #start = time.time()
            #print(f'{e}: read val: {start-end}')
            out = model.contact_map_prediction(tracks,
                                               seq,
                                               diag_init[:tracks.shape[0]])
            validation_corr.append(util.vec_corr_batch(
                diagonals[:, index_diagonal(data.start)[-1]:],
                out
            ).detach().cpu().numpy())
            validation_loss.append(loss(out, diagonals[:, index_diagonal(data.start)[-1]:]))
            #end = time.time()
            #print(f'{e}: inner val: {end-start}')

        mean_corr = np.mean(validation_corr)
        mean_val_loss = np.mean(validation_loss)
        mean_corr_traj_val.append(mean_corr)
        mean_loss_traj_val.append(mean_val_loss)

    if mean_corr > best_corr:
        best_corr = mean_corr
        torch.save(model.state_dict(),
                   os.path.join(SAVE_FOLDER, "best_correlation.pt"))

    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        torch.save(model.state_dict(),
                   os.path.join(SAVE_FOLDER, "best_val_lost.pt"))

    scheduler.step(mean_corr)
    if scheduler.get_last_lr()[-1] < LR_THRESHOLD:
        break

    np.save(os.path.join(SAVE_FOLDER, "mean_loss_traj_train.npy"), mean_loss_traj_train)
    np.save(os.path.join(SAVE_FOLDER, "mean_corr_traj_val.npy"), mean_corr_traj_val)
    np.save(os.path.join(SAVE_FOLDER, "mean_loss_traj_val.npy"), mean_loss_traj_val)

    print(f'{int((e+1)/NUM_EPOCH*100):3}/100: '
            f'correlation = {mean_corr:.3f}, '
            f'loss = {mean_total_loss:.3f}, '
            f'val_loss = {mean_val_loss:.3f}',
            flush=True, end='\r')

print(f'{int((e+1)/NUM_EPOCH*100):3}/100: '
            f'correlation = {mean_corr:.3f}, '
            f'loss = {mean_total_loss:.3f}, '
            f'val_loss = {mean_val_loss:.3f}')
