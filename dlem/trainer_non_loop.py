"""Contains Pytorch Lightning trainer class for DLEM.
"""
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import optim
from dlem import util

class LitTrainerStage2(L.LightningModule):
    """Trainer class for DLEM.
    """
    def __init__(self, model_loop,
                 model_non_loop,
                 learning_rate,
                 loss,
                 patch_dim,
                 start,
                 stop,
                 depth,
                 metric_file_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.model_loop = model_loop

        #freeze the loop model
        for param in self.model_loop.parameters():
            param.requires_grad = False

        self.model_non_loop = model_non_loop

        self.learning_rate = learning_rate
        self.loss = loss
        self.patch_dim = patch_dim
        self.start = start
        self.stop = stop
        self.index_diagonal = util.diag_index_for_mat(self.patch_dim, self.start, self.stop)
        self._all_the_metrics_val = dict()
        self._all_the_metrics_test = dict()
        self.training_loss = []
        self.metric_file_path = metric_file_path

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    def training_step(self, batch, _):
        """Training step for the model.
        """
        #if next(self.model.parameters()).device != self.device_model:
        #    self.model = self.model.to(self.device_model)

        seq, diagonals, tracks = batch
        diag_init = torch.from_numpy(np.ones((diagonals[0].shape[0], self.patch_dim - self.start),
                                             dtype=np.float32) * self.patch_dim)
        out = self.model_loop.contact_map_prediction(tracks,
                                                seq,
                                                diag_init[:tracks.shape[0]])
        out += self.model_non_loop(tracks.to(self.device)).cpu()
        diagonals = diagonals[:, self.index_diagonal(self.start)[-1]+1:].cpu()

        loss = self.loss(out, diagonals)
        self.training_loss.append(loss.detach().cpu().numpy())
        return loss

    def on_train_epoch_start(self):
        self.training_loss = []
        self.trainer.datamodule.on_train_epoch_start(self.current_epoch)

    def on_train_epoch_end(self):
        self.log("train_loss", np.mean(self.training_loss))
        return np.mean(self.training_loss)

    def evalution_step(self, batch:tuple) -> dict:
        """Evaluation step for the model. Used in validation and test steps.

        Args:
            batch (tuple): batch of data.

        Returns:
            dict: dictionary of metrics.
        """
        metrics = dict()
        seq, diagonals, tracks, names = batch

        diag_init = torch.from_numpy(np.ones((diagonals[0].shape[0], self.patch_dim - self.start),
                                             dtype=np.float32) * self.patch_dim)

        preds = []
        for diagonal, track, name in zip(diagonals, tracks, names):
            diagonal = diagonal[:, self.index_diagonal(self.start)[-1]+1:].cpu()
            out = self.model_loop.contact_map_prediction(track,
                                                    seq,
                                                    diag_init[:track.shape[0]])
            out += self.model_non_loop(track.to(self.device)).cpu()
            preds.append(out)
            corr = util.pairwise_corrcoef(out,
                                          diagonal)
            loss = self.loss(out, diagonal)

            metrics[f"corr_{name[0]}"] = corr

            metrics[f"loss_{name[0]}"] = loss

        diff_diag = diagonals[0][:, self.index_diagonal(self.start)[-1]+1:].cpu()
        diff_diag -= diagonals[1][:, self.index_diagonal(self.start)[-1]+1:].cpu()
        corr = util.pairwise_corrcoef(preds[0]-preds[1], diff_diag)
        metrics["corr_diff"] = corr

        return metrics

    def epoch_end_shared(self, metric_dict):
        for key in metric_dict:
            if "_loss_" in key:
                metric_dict[key] = torch.Tensor(
                    metric_dict[key]
                    )
            else:
                metric_dict[key] = torch.concat(
                    metric_dict[key]
                    )
        return metric_dict

    def test_step(self, batch, _):
        """Test step for the model.
        """
        metrics = self.evalution_step(batch)
        for key, val in metrics.items():
            if "test_" + key not in self._all_the_metrics_test:
                self._all_the_metrics_test["test_" + key] = [val]
            else:
                self._all_the_metrics_test["test_" + key].append(val)

    def on_test_epoch_start(self):
        self._all_the_metrics_test = dict()

    def on_test_epoch_end(self):
        self._all_the_metrics_test = self.epoch_end_shared(self._all_the_metrics_test)
        for key in self._all_the_metrics_test:
            self.log(
                key,
                self._all_the_metrics_test[key].mean(),
            )
        # Save the dictionary as a TSV file
        if self.metric_file_path is not None:
            metrics_df = pd.DataFrame.from_dict(dict(
                filter(lambda item: "corr" in item[0], self._all_the_metrics_test.items())
                ))
            metrics_df.to_csv(self.metric_file_path,
                                sep='\t',
                                index=False)

    def validation_step(self, batch, _):
        """Validation step for the model.
        """
        metrics = self.evalution_step(batch)
        for key, val in metrics.items():
            if "validation_" + key not in self._all_the_metrics_val:
                self._all_the_metrics_val["validation_" + key] = [val]
            else:
                self._all_the_metrics_val["validation_" + key].append(val)

    def on_validation_epoch_start(self):
        self._all_the_metrics_val = dict()

    def on_validation_epoch_end(self):
        self._all_the_metrics_val = self.epoch_end_shared(self._all_the_metrics_val)
        for key in self._all_the_metrics_val:
            self.log(
                key,
                self._all_the_metrics_val[key].mean(),
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model_non_loop.parameters(),
                               lr=self.learning_rate)
        return optimizer
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer, mode="max", patience=10, factor=0.5, verbose=True
        #)
        #return {
        #    "optimizer": optimizer,
        #    "lr_scheduler": {
        #        "scheduler": scheduler,
        #        "monitor": "validation_corr_H1",
        #        "frequency": 1,
        #    },
        #}
