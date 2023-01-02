# -*- coding: utf-8 -*-
import os
import time
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from pytorch_lightning import seed_everything
import warnings;warnings.simplefilter('ignore')

from utils import set_cpu_num
from utils.pnas_dataset import PNASDataset
from time_lagged_AE import TIME_LAGGED_AE


class FAST_EVOLVE(nn.Module):
    
    def __init__(self, in_channels, input_1d_width):
        super(FAST_EVOLVE, self).__init__()
                
        self.evolve_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*input_1d_width, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*input_1d_width, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (in_channels, input_1d_width))
        )
        
        def weights_normal_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                torch.nn.init.zeros_(m.bias)
        self.apply(weights_normal_init)

    def forward(self,x):
        
        out = self.evolve_mlp(x)
        return out


def train_fast_vars_evolve(tau, pretrain_epoch, is_print=False):
    
    assert tau > 0. # must time-lagged
    
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(tau)
    log_dir = f'logs/fast_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)
    os.makedirs(log_dir+"/slow_variable/", exist_ok=True)

    # init model
    time_lagged = TIME_LAGGED_AE(in_channels=1, input_1d_width=3, output_1d_width=3)
    fast_evol = FAST_EVOLVE(in_channels=1, input_1d_width=3).to(device)
    
    # load pretrained time-lagged AE
    ckpt_path = f'logs/time-lagged/tau_{tau}/checkpoints/epoch-{pretrain_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    time_lagged.load_state_dict(ckpt)
    time_lagged.eval()
    time_lagged = time_lagged.to('cpu')
    
    # training params
    lr = 0.01
    batch_size = 256
    max_epoch = 500
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(fast_evol.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()
    
    # dataset
    train_dataset = PNASDataset(data_filepath, 'train', 'MinMaxZeroOne',)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = PNASDataset(data_filepath, 'val', 'MinMaxZeroOne')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    
    # encode data into embedding
    train_batch = []
    val_batch = []
    with torch.no_grad():
        for input, target in train_loader:
            output, _ = time_lagged.forward(input.to(device)) # t0 --> t1
            fast_data = target - output
            train_batch.append(fast_data)
        for input, target in val_loader:
            output, _ = time_lagged.forward(input.to(device)) # t0 --> t1
            fast_data = target - output
            val_batch.append(fast_data)
    train_batch = [[train_batch[i], train_batch[i+1]] for i in range(len(train_batch)-1)]
    val_batch = [[val_batch[i], val_batch[i+1]] for i in range(len(val_batch)-1)]

    # training pipeline
    losses = []
    loss_curve = []
    best_mse = 1.
    for epoch in range(1, max_epoch+1):
        
        # train
        fast_evol.train()
        for batch in train_batch:
            input = batch[0]
            target = batch[1]
            
            output = fast_evol.forward(input)
            
            loss = loss_func(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().item())
            
        loss_curve.append(np.mean(losses))

        # validate
        with torch.no_grad():
            targets = []
            outputs = []
            
            fast_evol.eval()
            for batch in val_batch:
                input = batch[0]
                target = batch[1]
            
                output = fast_evol.forward(input.to(device))
                
                outputs.append(output.cpu().detach())
                targets.append(target.cpu().detach())

            targets = torch.concat(targets, axis=0)
            outputs = torch.concat(outputs, axis=0)
            mse = loss_func(outputs, targets)
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] val-MSE={mse}      ', end='')

            # record best model
            if mse < best_mse:
                best_mse = mse
                best_model = fast_evol.state_dict()
    
    # plot slow variable
    plt.figure(figsize=(19, 9))
    for index, item in enumerate(['X', 'Y', 'Z']):
        ax = plt.subplot(1,3,index+1)
        ax.set_title(item)
        plt.plot(targets[:,0,index], label='true')
        plt.plot(outputs[:,0,index], label='predict')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
    plt.savefig(log_dir+"/best_result.jpg", dpi=300)
    plt.close()
    
    # save model
    torch.save(best_model, log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    if is_print: print(f'\nsave best model at {log_dir}/checkpoints/epoch-{epoch}.ckpt (val_loss={best_mse})')
    
    # plot loss curve
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel('epoch')
    plt.title('Train MSELoss Curve')
    plt.savefig(log_dir+'/loss_curve.jpg', dpi=300)
    np.save(log_dir+'/loss_curve.npy', loss_curve)


def pipeline(tau, pretrain_epoch, is_print=False, random_seed=1):
    
    time.sleep(1.0)
    set_cpu_num(1)

    seed_everything(random_seed)
    train_fast_vars_evolve(tau, pretrain_epoch, is_print=is_print)


if __name__ == '__main__':

    subprocess = []
    
    for tau in [1.5]:
        for pretrain_epoch in [2, 30]:
            is_print = True if len(subprocess)==0 else False
            subprocess.append(Process(target=pipeline, args=(tau, pretrain_epoch, is_print), daemon=True))
            subprocess[-1].start()
    
    while any([subp.exitcode == None for subp in subprocess]):
        pass