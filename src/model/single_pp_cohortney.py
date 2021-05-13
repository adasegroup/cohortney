__all__ = ['SinglePointProcessSystem']
"""
This file contains system for training clustering with single point process model
"""
import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.dataset import dataset_dict
from src.networks.losses import cohortney_criterion
from src.utils import get_optimizer, get_scheduler, get_learning_rate


class SinglePointProcessSystem(pl.LightningModule):
    """
        Trainer for single point process model
    """

    def __init__(self, hparams):
        """
            :arg:
                  process - torch.nn.Module, model to train
                  optimizer - optimizer to train model
                  criterion - loss to optimize, takes batch, lambdas, dts
                  x - torch.Tensor, training data
                  val - torch.Tensor, validation data
                  max_epochs - int, number of epochs for sgd training
                  batch_size - int, batch size
                  generator_model - torch.nn.Module, true model, that was used for generation or None
            :return:
                  the same as inputs
        """
        super(SinglePointProcessSystem, self).__init__()
        self.model = self.create_process_model(hparams)
        self.criterion = cohortney_criterion
        self.batch_size = hparams.batch_size
        self.generator_model = creat_generator_model(hparams)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)
        self.N = len(self.train_dataset)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.num_workers,
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        lambdas = self.model(batch)
        loss = self.criterion(batch, lambdas, batch[:, 0, 0])

        # saving results
        # changed dict to simple variable
        mse = 0.0
        if self.generator_model:
            true_lambdas = self.generator_model(batch)
            mse = np.var((lambdas.detach().numpy() - true_lambdas.detach().numpy()))

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss.item(),)

        self.log('train/mse', mse, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        """
        :param batch_nb:
        :return:
          val_ll - torch.Tensor, size = (1), log likelihood on validation dataset
                   val_mse - float, mean squared error between obtained lambdas and lambdas of true model on validation,
                             if true model is not provided, then None
        """
        lambdas = self.model(batch)
        result = {}
        result['val_ll'] = self.criterion(self.val, lambdas, batch[:, 0, 0])
        if self.generator_model:
            true_lambdas = self.generator_model(batch)
            result['val_mse'] = np.var((lambdas.detach().numpy() - true_lambdas.detach().numpy()))
        return result

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_ll'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_mse'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)