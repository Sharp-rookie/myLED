# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything

from Data.dataset import PNASDataset4TCN
from Data.generator import generate_dataset


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.tanh = nn.Tanh()
        
        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, input_1d_width, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, input_1d_width, dtype=torch.float32))

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output).double()
        return self.tanh(output)
    
        # (batchsize,sequence_length,1,3)-->(batchsize,sequence_length,3)
        
        # (batchsize,sequence_length,3)-->(batchsize,3,sequence_length)
        
        # (batchsize,sequence_length,3)-->(batchsize,sequence_length,1,3)
        
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min


def train(tau, delta_t, sequence_length, is_print=False):
        
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
    batch_size = 32
    max_epoch = 50
    weight_decay = 0.001
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # dataset
    train_dataset = PNASDataset4TCN(data_filepath, 'train', length=sequence_length)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = PNASDataset4TCN(data_filepath, 'val', length=sequence_length)
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
    

def test_evolve(tau, ckpt_epoch, delta_t, n, sequence_length, is_print=False):
        
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/lstm/tau_{tau}'

    # load model
    batch_size = 32
    model = LSTM(in_channels=1, input_1d_width=3)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    # dataset
    test_dataset = PNASDataset4TCN(data_filepath, 'test', length=sequence_length+n)
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


def main(trace_num, tau, n, is_print=False):
    
    seed_everything(729)
    
    sample_num = None
    
    # TODO: 需要为TCN写一个getitem输出是sequence_length的dataset类，因为TCN输入必须是sequence，和我们的模型、LSTM的单样本输入不同
    # 因此，TCN的train可以和lstm的相似，仅训练单步；但是test时的外推，dataset和LSTM的不一样，需要是delta_t==tau/n，sequence_length=(n+外推时长/delta_t)

    # train
    generate_dataset(trace_num, round(tau/n, 3), sample_num, False, n)
    train(tau, round(tau/n,3), is_print=is_print)
    
    # test evolve
    ckpt_epoch = 50
    for i in range(1, n+1):
        delta_t = round(tau/n*i, 3)
        generate_dataset(trace_num, delta_t, sample_num, False, n+i)
        mse, mae = test_evolve(tau, ckpt_epoch, delta_t, i, sequence_length, is_print)
        with open('lstm_evolve_test.txt','a') as f:
            f.writelines(f'{delta_t}, {mse}, {mae}\n')


if __name__ == '__main__':
    
    trace_num = 128 + 16 + 16
    sequence_length = 10
    
    main(trace_num=trace_num, tau=4.5, n=sequence_length, is_print=True)