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

from utils.pnas_dataset import PNASDataset, scaler
from utils.pnas_model import PNAS_VisDynamicsModel
from utils.common import rm_mkdir


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
    
    def forward(self, x):
        
        x = x.squeeze()
        slow_variable = self.encoder(x)
        output = self.decoder(slow_variable)
        output = output.unsqueeze(-1)
        
        return output, slow_variable


def generate_input_and_embedding(tau, pretrain_epoch, data_filepath):
    
    if os.path.exists(f'Data/slow_AE/tau_{tau}/pretrain_epoch{pretrain_epoch}/test.npz'): return
    
    # prepare
    ckpt_path = f'logs/time-lagged/logs_tau{tau}_pnas_pnas-ae_1/lightning_logs/checkpoints/epoch={pretrain_epoch}.ckpt'
    os.makedirs(f'Data/slow_AE/tau_{tau}/pretrain_epoch{pretrain_epoch}', exist_ok=True)
    
    # load pretrained Time-lagged AE model
    TL_AE = PNAS_VisDynamicsModel(
        in_channels=1,
        input_1d_width=3,
        ouput_1d_width=3,
    )
    ckpt = torch.load(ckpt_path)
    TL_AE.load_state_dict(ckpt['state_dict'])
    TL_AE.eval().freeze()
    TL_AE = TL_AE.to('cpu')
    
    # Scaler
    input_scaler = scaler(
                        scaler_type='MinMaxZeroOne',
                        data_min=np.loadtxt(data_filepath+"/data_min.txt"),
                        data_max=np.loadtxt(data_filepath+"/data_max.txt"),)
    target_scaler = scaler(
                        scaler_type='MinMaxZeroOne',
                        data_min=np.loadtxt(data_filepath+"/data_min.txt"),
                        data_max=np.loadtxt(data_filepath+"/data_max.txt"))
    
    for item in ['train', 'val', 'test']:
        
        # 导入数据集
        dataset = PNASDataset(
                            file_path=data_filepath+f'/{item}.npz', 
                            input_scaler=input_scaler,
                            target_scaler=target_scaler)
        dataloader = torch.utils.data.DataLoader(
                            dataset=dataset,
                            batch_size=256,
                            shuffle=False)
        
        # 编码得到train、val、test的embedding
        inputs = []
        embeddings = []
        for input, _ in dataloader:
            _, embedding = TL_AE.model(input.to('cpu'))
            embeddings.append(embedding.cpu().detach().numpy())
            inputs.append(input.cpu().detach().numpy())
        
        # input和embedding对应存储在指定路径
        np.savez(f'Data/slow_AE/tau_{tau}/pretrain_epoch{pretrain_epoch}/{item}.npz', inputs=inputs, embeddings=embeddings)


def slow_ae_main(tau, pretrain_epoch, id):
    
    # prepare
    data_filepath = f'Data/slow_AE/tau_{tau}/pretrain_epoch{pretrain_epoch}'
    log_dir = f'logs/slow_ae/tau_{tau}/pretrain_epoch{pretrain_epoch}/id{id}/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"checkpoints/", exist_ok=True)
    os.makedirs(log_dir+"slow_variable/", exist_ok=True)
    
    # 导入数据
    train_data = np.load(data_filepath+'/train.npz', allow_pickle=True)['embeddings']
    val_data = np.load(data_filepath+'/val.npz', allow_pickle=True)['embeddings']
    val_inputs = np.load(data_filepath+'/val.npz', allow_pickle=True)['inputs']
    
    # 初始化model
    model = SLOW_AE(input_dim=64, slow_dim=id, output_dim=64)
    model.to(torch.device('cpu'))
    
    # 训练设置
    lr = 0.01
    max_epoch = 100
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()
    
    # 训练pipeline
    losses = []
    loss_curve = []
    best_val_loss = 1.
    for epoch in range(1, max_epoch+1):
        
        # train
        for input in train_data:
            input = torch.from_numpy(input)
            output, _ = model.forward(input)
            
            loss = loss_func(output, input) # reconstruction
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().item())
            
        loss_curve.append(np.mean(losses))

        # validate
        with torch.no_grad():
            # TODO: 变量重命名，避免歧义
            inputs = []
            embeddings = []
            outputs = []
            slow_variables = []
            for embedding, input in zip(val_data, val_inputs):
                input = torch.from_numpy(input)
                embedding = torch.from_numpy(embedding)
                output, slow = model.forward(embedding)
                inputs.append(input.cpu().detach())
                embeddings.append(embedding.cpu().detach())
                outputs.append(output.cpu().detach())
                slow_variables.append(slow.cpu().detach())
            inputs = torch.concat(inputs, axis=0)
            embeddings = torch.concat(embeddings, axis=0)
            outputs = torch.concat(outputs, axis=0)
            slow_variables = torch.concat(slow_variables, axis=0)
            
            mse = loss_func(outputs, embeddings)
            print(f'\rTau[{tau}] | ID[{id}] | Epoch[{epoch}] Val-MSE = {mse:.5f}', end='')
            
            # plot slow variable
            plt.figure(figsize=(12,4+2*id))
            for id_var in range(id):
                for index, item in enumerate(['X', 'Y', 'Z']):
                    plt.subplot(id,3,index+1+3*(id_var))
                    plt.scatter(inputs[:, 0, index], slow_variables[:, id_var])
                    plt.xlabel(item)
                    plt.ylabel(f'ID[{id_var+1}]')
            plt.subplots_adjust(wspace=0.35, hspace=0.35)
            plt.savefig(log_dir+f"slow_variable/epoch{epoch}.jpg", dpi=300)
            plt.close()
            
            # record best model
            if mse < best_val_loss:
                best_val_loss = mse
                best_model = model.state_dict()
    
    # save model
    torch.save(best_model, log_dir+f"checkpoints/epoch{epoch}_val_loss{mse:.5f}.ckpt")
    print(f'\nsave model(bet val loss: {mse:.5f})')
    
    # plot loss curve
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel('epoch')
    plt.title('Train MSELoss Curve')
    plt.savefig(log_dir+'loss_curve.jpg', dpi=300)
    np.save(log_dir+'loss_curve.npy', loss_curve)
    


def test_mse_and_pertinence():
    pass
    # 加载训练好的slow-AE模型
    # 加载测试集Dataloader
    # 测试mse
    # 绘制embedding与input-X、Y、Z各自的变化曲线


def pipeline(tau, pretrain_epoch, id):
    
    time.sleep(1.0)
    generate_input_and_embedding(tau, pretrain_epoch, f'Data/data/tau_{tau}')
    slow_ae_main(tau, pretrain_epoch, id)


if __name__ == '__main__':
    
    seed_everything(1)
    
    subprocess = []
    subprocess.append(Process(target=pipeline, args=(0.0, 6, 1), daemon=True))
    subprocess.append(Process(target=pipeline, args=(0.0, 6, 2), daemon=True))
    subprocess.append(Process(target=pipeline, args=(0.0, 6, 3), daemon=True))
    subprocess.append(Process(target=pipeline, args=(0.0, 6, 4), daemon=True))
    subprocess.append(Process(target=pipeline, args=(0.0, 6, 5), daemon=True))
    subprocess.append(Process(target=pipeline, args=(0.0, 30, 1), daemon=True))
    subprocess.append(Process(target=pipeline, args=(0.0, 30, 2), daemon=True))
    subprocess.append(Process(target=pipeline, args=(0.0, 30, 3), daemon=True))
    subprocess.append(Process(target=pipeline, args=(0.0, 30, 4), daemon=True))
    subprocess.append(Process(target=pipeline, args=(0.0, 30, 5), daemon=True))
    [subp.start() for subp in subprocess]
    
    while any([subp.exitcode == None for subp in subprocess]):
        pass