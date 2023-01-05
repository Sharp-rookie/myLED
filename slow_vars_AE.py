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


class SLOW_AE(nn.Module):
    
    def __init__(self, input_dim, slow_dim, output_dim):
        super(SLOW_AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, slow_dim, bias=True),
            nn.Tanh(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(slow_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, output_dim, bias=True),
            nn.Tanh(),
        )
        
        def weights_normal_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)
        self.apply(weights_normal_init)
    
    def forward(self, x):
        
        x = x.squeeze()
        slow_variable = self.encoder(x)
        output = self.decoder(slow_variable)
        output = output.unsqueeze(-1)
        
        return output, slow_variable


def generate_input_and_embedding(tau, pretrain_epoch, data_filepath):
    
    if os.path.exists(f'Data/slow_AE/tau_{tau}/pretrain_epoch{pretrain_epoch}/test.npz'): return
    
    # prepare
    ckpt_path = f'logs/time-lagged/tau_{tau}/checkpoints/epoch-{pretrain_epoch}.ckpt'
    os.makedirs(f'Data/slow_AE/tau_{tau}/pretrain_epoch{pretrain_epoch}', exist_ok=True)
    
    # load pretrained Time-lagged AE model
    TL_AE = TIME_LAGGED_AE(in_channels=1, input_1d_width=3, output_1d_width=3,)
    ckpt = torch.load(ckpt_path)
    TL_AE.load_state_dict(ckpt)
    TL_AE.eval()
    TL_AE = TL_AE.to('cpu')
    
    for item in ['train', 'val', 'test']:
        
        # 导入数据集
        dataset = PNASDataset(data_filepath, item, 'MinMaxZeroOne',)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=False)
        
        # 编码得到train、val、test的embedding
        inputs = []
        embeddings = []
        with torch.no_grad():
            for input, _ in dataloader:
                _, embedding = TL_AE.forward(input.to('cpu'))
                embeddings.append(embedding.cpu().detach().numpy())
                inputs.append(input.cpu().detach().numpy())
        
        # input和embedding对应存储在指定路径
        np.savez(f'Data/slow_AE/tau_{tau}/pretrain_epoch{pretrain_epoch}/{item}.npz', inputs=inputs, embeddings=embeddings)


def slow_ae_main(tau, pretrain_epoch, id, is_print=False):
    
    # prepare
    data_filepath = f'Data/slow_AE/tau_{tau}/pretrain_epoch{pretrain_epoch}'
    log_dir = f'logs/slow_ae/tau_{tau}/pretrain_epoch{pretrain_epoch}/id{id}/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"checkpoints/", exist_ok=True)
    os.makedirs(log_dir+"slow_variable/", exist_ok=True)

    # 初始化model
    model = SLOW_AE(input_dim=64, slow_dim=id, output_dim=64)
    model.to(torch.device('cpu'))
    
    # 训练设置
    lr = 0.01
    max_epoch = 100
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()
    
    # 导入数据
    train_embeddings = np.load(data_filepath+'/train.npz', allow_pickle=True)['embeddings']
    val_embeddings = np.load(data_filepath+'/val.npz', allow_pickle=True)['embeddings']
    val_inputs = np.load(data_filepath+'/val.npz', allow_pickle=True)['inputs']
    
    # 训练pipeline
    losses = []
    loss_curve = []
    best_val_loss = 1.
    for epoch in range(1, max_epoch+1):
        
        # train
        model.train()
        for embedding in train_embeddings:

            embedding = torch.from_numpy(embedding)

            reconstruct_embedding, _ = model.forward(embedding)
            
            loss = loss_func(reconstruct_embedding, embedding)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().item())
            
        loss_curve.append(np.mean(losses))

        # validate
        with torch.no_grad():
            
            inputs = []
            embeddings = []
            reconstruct_embeddings = []
            slow_variables = []
            
            model.eval()
            for embedding, input in zip(val_embeddings, val_inputs):

                input = torch.from_numpy(input)
                embedding = torch.from_numpy(embedding)

                reconstruct_embedding, slow = model.forward(embedding)

                inputs.append(input.cpu())
                reconstruct_embeddings.append(reconstruct_embedding.cpu())
                embeddings.append(embedding.cpu())
                slow_variables.append(slow.cpu())
                
            inputs = torch.concat(inputs, axis=0)
            embeddings = torch.concat(embeddings, axis=0)
            reconstruct_embeddings = torch.concat(reconstruct_embeddings, axis=0)
            slow_variables = torch.concat(slow_variables, axis=0)
            
            mse = loss_func(reconstruct_embeddings, embeddings)
            if is_print: print(f'\rTau[{tau}] | ID[{id}] | epoch[{epoch}/{max_epoch}] Val-MSE={mse:.5f}', end='')
            
            
            # plot slow variable
            plt.figure(figsize=(12,4+2*id))
            for id_var in range(id):
                for index, item in enumerate(['X', 'Y', 'Z']):
                    plt.subplot(id,3,index+1+3*(id_var))
                    plt.scatter(inputs[:, 0, index], slow_variables[:, id_var])
                    plt.xlabel(item)
                    plt.ylabel(f'ID[{id_var+1}]')
            plt.subplots_adjust(wspace=0.35, hspace=0.35)
            plt.savefig(log_dir+f"slow_variable/epoch-{epoch}.jpg", dpi=300)
            plt.close()
            
            # record best model
            if mse < best_val_loss:
                best_val_loss = mse
                best_model = model.state_dict()
    
    # save model
    torch.save(best_model, log_dir+f"checkpoints/epoch-{epoch}_val-mse-{mse:.5f}.ckpt")
    if is_print: print(f'\nsave model(bet val loss: {mse:.5f})')
    
    # plot loss curve
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel('epoch')
    plt.title('Train MSELoss Curve')
    plt.savefig(log_dir+'loss_curve.jpg', dpi=300)
    np.save(log_dir+'loss_curve.npy', loss_curve)


def pipeline(tau, pretrain_epoch, id, is_print=False, random_seed=1):
    
    time.sleep(1.0)
    set_cpu_num(1)

    seed_everything(random_seed)
    slow_ae_main(tau, pretrain_epoch, id, is_print=is_print)


if __name__ == '__main__':

    subprocess = []
    
    for tau in [0.0]:
        for pretrain_epoch in [4, 20]:

            generate_input_and_embedding(tau, pretrain_epoch, f'Data/data/tau_{tau}')

            for id in [1,2,3,4]:
                is_print = True if len(subprocess)==0 else False
                subprocess.append(Process(target=pipeline, args=(tau, pretrain_epoch, id, is_print), daemon=True))
                subprocess[-1].start()
    
    while any([subp.exitcode == None for subp in subprocess]):
        pass