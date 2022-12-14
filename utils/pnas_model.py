import os
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl

from .pnas_ae_nets import Cnov_AE, MLP_AE
from .pnas_dataset import PNASDataset, scaler


class PNAS_VisDynamicsModel(pl.LightningModule):

    def __init__(self,
                 lr: float=1e-4,
                 seed: int=1,
                 if_cuda: bool=True,
                 if_test: bool=False,
                 gamma: float=0.5,
                 log_dir: str='logs',
                 train_batch: int=256,
                 val_batch: int=256,
                 test_batch: int=256,
                 num_workers: int=8,
                 pin_memory: bool=True,
                 model_name: str='encoder-decoder-64',
                 data_filepath: str='data',
                 dataset: str='single_pendulum',
                 lr_schedule: list=[20, 50, 100],
                 in_channels: int=2,
                 input_1d_width: int=101,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': self.hparams.pin_memory} if self.hparams.if_cuda else {}

        self.__build_model()

    def __build_model(self):
        
        # model
        self.model = Cnov_AE(in_channels=self.hparams.in_channels, input_1d_width=self.hparams.input_1d_width)
        # self.model = MLP_AE(in_channels=self.hparams.in_channels, input_1d_width=self.hparams.input_1d_width)
        
        # loss
        self.loss_func = nn.MSELoss()

    def train_forward(self, x):

        output, latent = self.model(x)
        return output, latent

    def training_step(self, batch, batch_idx):
        
        data, target = batch
        output, _ = self.train_forward(data)
        train_loss = self.loss_func(output, target)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        
        data, target = batch
        output, _ = self.train_forward(data)
        val_loss = self.loss_func(output, target)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        
        data, target = batch
        output, latent = self.model(data, data)
        test_loss = self.loss_func(output, target)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # save the latent vectors
        for idx in range(data.shape[0]):
            latent_tmp = latent[idx].view(1, -1)[0]
            latent_tmp = latent_tmp.cpu().detach().numpy()
            self.all_latents.append(latent_tmp)

    def test_save(self):
        
        np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
    
    def paths_to_tuple(self, paths):
        new_paths = []
        for i in range(len(paths)):
            tmp = paths[i].split('.')[0].split('_')
            new_paths.append((int(tmp[0]), int(tmp[1])))
        return new_paths

    def setup(self, stage=None):

        self.input_scaler = scaler(
                            scaler_type='MinMaxZeroOne',
                            data_min=np.loadtxt(self.hparams.data_filepath+"/data_min.txt"),
                            data_max=np.loadtxt(self.hparams.data_filepath+"/data_max.txt"),
                )
        self.target_scaler = scaler(
                            scaler_type='MinMaxZeroOne',
                            data_min=np.loadtxt(self.hparams.data_filepath+"/data_min.txt"),
                            data_max=np.loadtxt(self.hparams.data_filepath+"/data_max.txt"),
                )

        if stage == 'fit':
            self.train_dataset = PNASDataset(
                file_path=self.hparams.data_filepath+'/train.npz', 
                input_scaler=self.input_scaler,
                target_scaler=self.target_scaler,
                )
            self.val_dataset = PNASDataset(
                file_path=self.hparams.data_filepath+'/val.npz', 
                input_scaler=self.input_scaler,
                target_scaler=self.target_scaler,
                )

        if stage == 'test':
            self.test_dataset = PNASDataset(
                file_path=self.hparams.data_filepath+'/test.npz', 
                input_scaler=self.input_scaler,
                target_scaler=self.target_scaler,
                )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.hparams.train_batch,
                                                   shuffle=True,
                                                   **self.kwargs)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                 batch_size=self.hparams.val_batch,
                                                 shuffle=False,
                                                 **self.kwargs)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.hparams.test_batch,
                                                  shuffle=False,
                                                  **self.kwargs)
        return test_loader