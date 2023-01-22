# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything

from Data.dataset import PNASDataset
    
    
class LSTM(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, hidden_dim=64, layer_num=2):
        super(LSTM, self).__init__()
        
        # (batchsize,1,1,3)-->(batchsize,1,3)
        self.flatten = nn.Flatten(start_dim=2)
        
        # (batchsize,1,3)-->(batchsize, hidden_dim)
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(
            input_size=in_channels*input_1d_width, 
            hidden_size=hidden_dim, 
            num_layers=layer_num, 
            dropout=0.01, 
            batch_first=True # input: (batch_size, squences, features)
            )
        
        # (batchsize, hidden_dim)-->(batchsize, 3)
        self.fc = nn.Linear(hidden_dim, in_channels*input_1d_width)
        
        # (batchsize, 3)-->(batchsize,1,1,3)
        self.unflatten = nn.Unflatten(-1, (1, in_channels, input_1d_width))

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, input_1d_width, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, input_1d_width, dtype=torch.float32))
    
    def forward(self, x):
        
        h0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32)
        c0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32)
        
        x = self.flatten(x)
        _, (h, c)  = self.cell(x, (h0, c0))
        y = self.fc(h[-1])
        y = self.unflatten(y)
        
        return y
    
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min


def train(tau, delta_t, is_print=False):
        
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/lstm/tau_{tau}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    model = LSTM(in_channels=1, input_1d_width=3)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    
    # training params
    lr = 0.001
    batch_size = 128
    max_epoch = 50
    weight_decay = 0.001
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # dataset
    train_dataset = PNASDataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = PNASDataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    train_loss = []
    best_loss = 1.
    for epoch in range(1, max_epoch+1):
        
        losses = []
        
        # train
        model.train()
        for input, target in train_loader:
            
            input = model.scale(input).to(device) # (batchsize,1,1,3)
            target = model.scale(target).to(device)
            
            output = model(input)
            loss = MSE_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # record loss
            losses.append(loss.detach().item())
            
        train_loss.append(np.mean(losses[0]))
        
        # validate
        with torch.no_grad():
            targets = []
            outputs = []
            
            model.eval()
            for input, target in val_loader:
                
                input = model.scale(input).to(device) # (batchsize,1,1,3)
                target = model.scale(target).to(device)
                
                output = model(input)

                # record results
                outputs.append(output.cpu())
                targets.append(target.cpu())
            
            # trans to tensor
            outputs = torch.concat(outputs, axis=0)
            targets = torch.concat(targets, axis=0)
            
            # cal loss
            loss = MSE_loss(outputs, targets)
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] | val-mse={loss:.5f}', end='')
                        
            # plot per 5 epoch
            if epoch % 5 == 0:
                
                os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)

                period_num = 5*int(9/delta_t)
                
                # plot total infomation one-step prediction curve
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['X','Y','Z']):
                    ax = plt.subplot(1,3,j+1)
                    ax.set_title(item)
                    plt.plot(targets[:period_num,0,0,j], label='true')
                    plt.plot(outputs[:period_num,0,0,j], label='predict')
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/predict.jpg", dpi=300)
                plt.close()
        
            # record best model
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()

    # save model
    torch.save(best_model, log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    if is_print: print(f'\nsave best model at {log_dir}/checkpoints/epoch-{epoch}.ckpt (val best_loss={best_loss})')
    
    # plot loss curve
    train_loss = np.array(train_loss)
    plt.figure()
    plt.plot(train_loss)
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.savefig(log_dir+'/train_loss_curve.jpg', dpi=300)
    

def test_evolve(tau, ckpt_epoch, delta_t, n, is_print=False):
        
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/lstm/tau_{tau}'

    # load model
    batch_size = 128
    model = LSTM(in_channels=1, input_1d_width=3)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    # dataset
    test_dataset = PNASDataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # testing pipeline        
    with torch.no_grad():
        targets = []
        outputs = []
        
        model.eval()
        for input, target in test_loader:
            input = model.scale(input)
            target = model.scale(target)

            output = model(input)
            for _ in range(1, n):
                output = model(output)

            targets.append(target.cpu())
            outputs.append(output.cpu())
        
        targets = model.descale(torch.concat(targets, axis=0))
        outputs = model.descale(torch.concat(outputs, axis=0))
        
        os.makedirs(log_dir+f"/test/{delta_t}/", exist_ok=True)
        
        sample_num = 50
        period_num = sample_num
        
        # plot total infomation prediction curve
        plt.figure(figsize=(16,5))
        for j, item in enumerate(['X','Y','Z']):
            ax = plt.subplot(1,3,j+1)
            ax.set_title(item)
            plt.plot(targets[:period_num,0,0,j], label='true')
            plt.plot(outputs[:period_num,0,0,j], label='predict')
        plt.subplots_adjust(wspace=0.2)
        plt.savefig(log_dir+f"/test/predict_{delta_t}.jpg", dpi=300)
        plt.close()
    
    return nn.MSELoss()(outputs, targets).item(), nn.L1Loss()(outputs, targets).item()


def main(tau, n, is_print=False):
    
    seed_everything(729)

    ckpt_epoch = 50

    # train
    train(tau, round(tau/n,3), is_print=is_print)
    # test evolve
    for i in range(1, n+1):
        delta_t = round(tau/n*i, 3)
        mse, mae = test_evolve(tau, ckpt_epoch, delta_t, i, is_print)
        with open('lstm_evolve_test.txt','a') as f:
            f.writelines(f'{delta_t}, {mse}, {mae}\n')


if __name__ == '__main__':
    
    main(tau=4.5, n=10, is_print=True)