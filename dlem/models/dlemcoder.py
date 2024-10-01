import torch
from typing import Tuple
from numpy.typing import ArrayLike
from dlem.util import get_diags

def dlem(left, right, curr_diag, index_diag, n, transform=True):

    diag_len = n - index_diag

    index_in_left = range(index_diag, n-1)
    index_in_right = range(1, n-index_diag)
    index_out_left = range(index_diag+1, n)
    index_out_right = range(0, n-index_diag-1)

    index_curr_diag_left = range(0, diag_len-1)
    index_curr_diag_right = range(1, diag_len)

    mass_in = curr_diag[:, index_curr_diag_right] * right[:, index_in_right]
    mass_in += curr_diag[:, index_curr_diag_left] * left[:, index_in_left]

    mass_out = right[:, index_out_right] + left[:, index_out_left]

    next_diag_pred = mass_in / mass_out

    if transform:
        next_diag_pred = torch.log(next_diag_pred)
        next_diag_pred = next_diag_pred - torch.mean(next_diag_pred)

    return next_diag_pred


class DLEM(torch.nn.Module):
    def __init__(self, n):
        super(DLEM, self).__init__()
        # Define your layers here

        self.channel_num = 10 

        self.scan_left = torch.nn.Conv2d(in_channels=1,
                                         out_channels=self.channel_num//2,
                                         kernel_size=(3,n),
                                         padding=(1,0))
        self.scan_right = torch.nn.Conv2d(in_channels=1,
                                          out_channels=self.channel_num//2,
                                          kernel_size=(n,3),
                                          padding=(0,1))

        self.conv = torch.nn.Sequential(torch.nn.Conv1d(in_channels=self.channel_num,
                                                        out_channels=self.channel_num,
                                                        kernel_size=3,
                                                        padding=1),
                                        torch.nn.ReLU(),
                                       *[torch.nn.Conv1d(in_channels=self.channel_num,
                                                         out_channels=self.channel_num,
                                                         kernel_size=3,
                                                         padding=1),
                                         torch.nn.ReLU()]*2,
                                       *[torch.nn.ConvTranspose1d(in_channels=self.channel_num,
                                                                  out_channels=self.channel_num,
                                                                  kernel_size=3,
                                                                  padding=1,
                                                                  output_padding=0),
                                         torch.nn.ReLU()]*3)

        self.mix_lr = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.channel_num,
                            out_channels=2,
                            kernel_size=1),
            torch.nn.Sigmoid())

    def forward(self, x, start, stop, transform=True):
        left_ = self.scan_left(x).squeeze()
        right_ = self.scan_right(x).squeeze()
        lr = torch.concat([left_, right_], axis=1)
        lr = self.conv(lr)
        lr = self.mix_lr(lr)
        left = lr[:, 0, :]
        right = lr[:, 1, :]
        preds = []
        for index_diag in range(start, stop):
            next_diag_pred = dlem(left, right, torch.exp(get_diags(x.squeeze(), index_diag)), index_diag, x.shape[-1], transform)
            preds.append(next_diag_pred)
        return preds 
