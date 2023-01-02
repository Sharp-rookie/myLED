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


class K_OPT(nn.Module):

    def __init__(self, id):
        super(K_OPT, self).__init__()

        self.Knet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(id, id, bias=True),
            nn.Unflatten(-1, (id, 1))
        )
    
    def forward(self, x):
        
        return self.Knet(x)


class SLOW_AE(nn.Module):
    
    def __init__(self, input_dim, slow_dim, output_dim):
        super(SLOW_AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, slow_dim, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (slow_dim, 1))
        )
        
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(slow_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, output_dim, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (output_dim, 1))
        )
        
        def weights_normal_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                torch.nn.init.zeros_(m.bias)
        self.apply(weights_normal_init)
    
    def forward(self, x):
        
        x = x.squeeze()
        slow_variable = self.encoder(x)
        output = self.decoder(slow_variable)
        output = output.unsqueeze(-1)
        
        return output, slow_variable


def e2e_train_slow_ae_and_knet(tau, pretrain_epoch, id, is_print=False):
    
    assert tau > 0. # must time-lagged
    
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(tau)
    log_dir = f'logs/slow_vars_koopman/tau_{tau}/pretrain_epoch{pretrain_epoch}/id{id}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)
    os.makedirs(log_dir+"/slow_variable/", exist_ok=True)

    # init model
    time_lagged = TIME_LAGGED_AE(in_channels=1, input_1d_width=3, output_1d_width=3)
    slow_ae = SLOW_AE(input_dim=64, slow_dim=id, output_dim=64).to(device)
    koopman = K_OPT(id=id).to(device)
    
    # load pretrained time-lagged AE
    ckpt_path = f'logs/time-lagged/tau_{tau}/checkpoints/epoch-{pretrain_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    time_lagged.load_state_dict(ckpt)
    time_lagged.eval()
    time_lagged = time_lagged.to('cpu')
    
    # training params
    lr = 0.01
    batch_size = 256
    max_epoch = 300
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(
        [{'params': slow_ae.parameters()}, 
         {'params': koopman.parameters()}],
        lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()
    
    # dataset
    train_dataset = PNASDataset(data_filepath, 'train', 'MinMaxZeroOne',)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = PNASDataset(data_filepath, 'val', 'MinMaxZeroOne')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    # encode data into embedding
    train_batch = []
    val_batch = []
    with torch.no_grad():
        for input, target in train_loader:
            _, in_embedding = time_lagged.forward(input.to(device))
            _, tar_embedding = time_lagged.forward(target.to(device))
            train_batch.append([in_embedding, tar_embedding, input, target])
        for input, target in val_loader:
            _, in_embedding = time_lagged.forward(input.to(device))
            _, tar_embedding = time_lagged.forward(target.to(device))
            val_batch.append([in_embedding, tar_embedding, input, target])

    # training pipeline
    losses = []
    loss_curve = []
    best_mse = 1.
    for epoch in range(1, max_epoch+1):
        
        # train
        slow_ae.train()
        koopman.train()
        for batch in train_batch:
            in_embedding = batch[0]
            tar_embedding = batch[1]
            
            slow_var_0 = slow_ae.encoder(in_embedding)
            slow_var_1 = koopman.forward(slow_var_0)
            output = slow_ae.decoder(slow_var_1)
            
            loss = loss_func(output, tar_embedding)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().item())
            
        loss_curve.append(np.mean(losses))

        # validate
        with torch.no_grad():
            inputs = []
            tar_embeddings = []
            out_embeddings = []
            slow_vars = []
            
            slow_ae.eval()
            koopman.eval()
            for batch in train_batch:
                in_embedding = batch[0]
                tar_embedding = batch[1]
                input = batch[2]
                target = batch[3]
                
                slow_var_t0 = slow_ae.encoder(in_embedding)
                slow_var_t1 = koopman.forward(slow_var_t0)
                out_embedding = slow_ae.decoder(slow_var_t1)
                
                inputs.append(input.cpu())
                tar_embeddings.append(tar_embedding.cpu())
                out_embeddings.append(out_embedding.cpu())
                slow_vars.append(slow_var_t0.cpu())
            
            inputs = torch.concat(inputs, axis=0)
            tar_embeddings = torch.concat(tar_embeddings, axis=0)
            out_embeddings = torch.concat(out_embeddings, axis=0)
            slow_vars = torch.concat(slow_vars, axis=0)
            
            embedding_mse = loss_func(out_embeddings, tar_embeddings)
            if is_print: print(f'\rTau[{tau}] | ID[{id}] | epoch[{epoch}/{max_epoch}] val: embedding-mse={embedding_mse}         ', end='')
                
            # plot slow variable
            plt.figure(figsize=(12,4+2*id))
            for id_var in range(id):
                for index, item in enumerate(['X', 'Y', 'Z']):
                    plt.subplot(id,3,index+1+3*(id_var))
                    plt.scatter(inputs[:, 0, index], slow_vars[:, id_var], s=5)
                    plt.xlabel(item)
                    plt.ylabel(f'ID[{id_var+1}]')
            plt.subplots_adjust(wspace=0.35, hspace=0.35)
            plt.savefig(log_dir+f"/slow_variable/epoch-{epoch}.jpg", dpi=300)
            plt.close()
            
            # record best model
            if embedding_mse < best_mse:
                best_mse = embedding_mse
                best_model = {'slow_ae': slow_ae.state_dict(),
                              'koopman': koopman.state_dict()}
    
    # save model
    torch.save(best_model, log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    if is_print: print(f'\nsave best model at {log_dir}/checkpoints/epoch-{epoch}.ckpt (val_loss={embedding_mse})')
    
    # plot loss curve
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel('epoch')
    plt.title('Train MSELoss Curve')
    plt.savefig(log_dir+'/loss_curve.jpg', dpi=300)
    np.save(log_dir+'/loss_curve.npy', loss_curve)


def pipeline(tau, pretrain_epoch, id, is_print=False, random_seed=1):
    
    time.sleep(1.0)
    set_cpu_num(1)

    seed_everything(random_seed)
    e2e_train_slow_ae_and_knet(tau, pretrain_epoch, id, is_print=is_print)


if __name__ == '__main__':

    subprocess = []
    
    for tau in [1.5]:
        for pretrain_epoch in [2, 30]:
            for id in [1, 2, 3, 4]:
                is_print = True if len(subprocess)==0 else False
                subprocess.append(Process(target=pipeline, args=(tau, pretrain_epoch, id, is_print), daemon=True))
                subprocess[-1].start()
    
    while any([subp.exitcode == None for subp in subprocess]):
        pass