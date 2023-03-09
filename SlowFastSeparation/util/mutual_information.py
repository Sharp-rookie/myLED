import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from multiprocessing import Process
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
from warnings import simplefilter;simplefilter('ignore')

from .common import set_cpu_num


def weight_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight,std=0.02)
        nn.init.constant_(layer.bias, 0)

class Mine(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.apply(weight_init)
        
    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint))
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
    
    # unbiasing use moving average
    loss = -(torch.mean(t) - torch.mean(et))/(ma_et.mean().detach())
    # use biased estimator
    # loss = - mi_lb
    
    mine_net_optim.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mi_lb, ma_et


def sample_batch(data, dim, batch_size=100, sample_mode='joint'):
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([data[joint_index][:,:dim], data[marginal_index][:,dim:]], axis=1)
    return batch


def train(data, dim, mine_net, mine_net_optim, batch_size=128, max_iters=1024, is_print=False):
    # data is x or y
    result = list()
    ma_et = 1.
    iter = tqdm(range(max_iters)) if is_print else range(max_iters)
    for _ in iter:
        batch = sample_batch(data, dim, batch_size), sample_batch(data, dim, batch_size, sample_mode='marginal')
        mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())

    return result


def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0, len(a)-window_size)]


def cal_mi_xy(x, y, dim, max_iters=1024, is_print=False):

    assert len(x.shape)==2 and len(y.shape)==2, f"Shape of x and y must be 2-d array: x--{x.shape}, y--{y.shape}"
    data = np.concatenate([x, y], axis=1)
    mine_net = Mine(input_size=2*dim)
    mine_net_optim = torch.optim.Adam(mine_net.parameters(), lr=1e-3)
    mi_per_iter = train(data, dim, mine_net, mine_net_optim, max_iters=max_iters, is_print=is_print)

    return mi_per_iter


def cal_mi_of_tau(tau, system, obs_dim, data_dim, result_filepath, max_iters=1024, is_print=False):

    time.sleep(0.1)
    set_cpu_num(1)
    
    data_path = f'Data/{system}/data/tau_{round(tau,2)}/test.npz'

    single_var_mi_string = ''
    obs_mi, hidden_mi = 0., 0.
    try:
        data = np.load(data_path)['data'].squeeze()

        # combined vector(observated variables) MI
        if tau == 0.0:
            x0 = data[:, :obs_dim]
            x1 = data[:, :obs_dim]
        else:
            x0 = data[:, 0, :obs_dim]
            x1 = data[:, 1, :obs_dim]
        obs_mi = cal_mi_xy(x0, x1, obs_dim, max_iters=max_iters, is_print=is_print)

        # combined vector(hidden variables, if exist) MI
        if obs_dim < data_dim:
            if tau == 0.0:
                x0 = data[:, obs_dim:]
                x1 = x0
            else:
                x0 = data[:, 0, obs_dim:]
                x1 = data[:, 1, obs_dim:]
            hidden_mi = cal_mi_xy(x0, x1, data_dim-obs_dim, max_iters=max_iters, is_print=is_print)

        # single variable self-MI
        for i in range(data_dim):
            if tau == 0.0:
                x0 = data[:, i:i+1]
                x1 = x0
            else:
                x0 = data[:, 0, i:i+1]
                x1 = data[:, 1, i:i+1]
            single_var_mi = cal_mi_xy(x0, x1, 1, max_iters=max_iters, is_print=is_print)
            single_var_mi_string += str(np.mean(single_var_mi[-50:])) + (',' if i<data_dim-1 else '')
        
        # single variable inter-MI
    
    except FileNotFoundError:
        pass
    
    with open(result_filepath, 'a') as fp:
        if data_dim > obs_dim:
            fp.writelines(f'{tau:.2f},{np.mean(obs_mi[-50:])},{np.mean(hidden_mi[-50:])},{single_var_mi_string}\n')
        else:
            fp.writelines(f'{tau:.2f},{np.mean(obs_mi[-50:])},{0},{single_var_mi_string}\n')


def cal_mi_system(tau_list, system, obs_dim, data_dim, max_iters=1024, parallel=False):

    os.makedirs(f'Data/{system}/mi/', exist_ok=True)
    result_filepath = f'Data/{system}/mi/mi.txt'
    if os.path.exists(result_filepath): return

    workers = []
    for tau in tau_list:
        if parallel:
            is_print = True if len(workers)==0 else False
            workers.append(Process(target=cal_mi_of_tau, args=(tau, system, obs_dim, data_dim, result_filepath, max_iters, is_print), daemon=True))
            workers[-1].start()
        else:
            cal_mi_of_tau(tau, system, obs_dim, data_dim, result_filepath, max_iters, is_print=True)
    
    while parallel and any([sub.exitcode==None for sub in workers]):
        pass


def plot_mi(tau_list, obs_dim, data_dim, system):

    result_filepath = f'Data/{system}/mi/mi.txt'
    if os.path.exists(f'Data/{system}/mi/mi.pdf') and os.path.exists(f'Data/{system}/mi/mi_single.pdf'): return

    single_var_mi_list = [[] for _ in range(data_dim)]
    tau_list, obs_mi_list, hidden_mi_list = [], [], []
    with open(result_filepath, 'r') as fp:
        for line in fp.readlines():
            data = line.split(',')
            tau_list.append(float(data[0]))
            obs_mi_list.append(float(data[1]))
            hidden_mi_list.append(float(data[2]))
            for i in range(data_dim):
                single_var_mi_list[i].append(float(data[3+i]))
        
        index = np.argsort(tau_list)
        tau_list = np.array(tau_list)[index]
        obs_mi_list = np.array(obs_mi_list)[index]
        hidden_mi_list = np.array(hidden_mi_list)[index]
        single_var_mi_list = np.array(single_var_mi_list)[:, index]
    
    plt.figure(figsize=(6,6))
    plt.plot(tau_list, hidden_mi_list, label='combination')
    for i in range(data_dim-obs_dim): 
        plt.plot(tau_list, single_var_mi_list[i+obs_dim], label=f'var_{i+obs_dim}')
    plt.legend()
    plt.xlabel(r'$t/s$', fontsize=18)
    plt.ylabel('Mutual Information', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Hidden Variables')
    plt.savefig(f'Data/{system}/mi/hidden_mi.pdf', dpi=300)

    plt.figure(figsize=(6,6))
    plt.plot(tau_list, obs_mi_list, label='combination')
    for i in range(obs_dim): 
        plt.plot(tau_list, single_var_mi_list[i], label=f'var_{i}')
    plt.legend()
    plt.xlabel(r'$t/s$', fontsize=18)
    plt.ylabel('Mutual Information', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Observable Variables')
    plt.savefig(f'Data/{system}/mi/obs_mi.pdf', dpi=300)
