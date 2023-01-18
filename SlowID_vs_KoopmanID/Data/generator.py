import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings;warnings.simplefilter('ignore')

from scipy.stats import special_ortho_group


class Koopman_System(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, koopman_dim, tau_0=1):
        super(Koopman_System, self).__init__()
        
        assert in_channels*input_1d_width >= koopman_dim and koopman_dim > 1
        
        self.t = tau_0
        self.in_channels = in_channels
        self.input_1d_width = input_1d_width
        
        # observation matrix 'D'
        self.D = torch.from_numpy(special_ortho_group.rvs(koopman_dim)).float()
        for _ in range(in_channels*input_1d_width-koopman_dim):
            self.D = torch.concat([self.D, 2*(torch.rand(1,koopman_dim)-0.5)], dim=0)
        
        # K matrix
        Lambda = -torch.rand(koopman_dim).sort().values
        Lambda = Lambda / 10 # 强制接近0，全为慢变量
        self.K_opt = torch.diag(torch.exp(tau_0 * Lambda))
        
        # record
        with open(f'Data/origin/k_{koopman_dim}/koopman_param.txt', 'w') as f:
            f.writelines(str(self.D.numpy()) + '\n')
            f.writelines(str(Lambda.numpy()) + '\n')
            f.writelines(str(self.K_opt.numpy()) + '\n')
    
    def k_evol(self, var):
        return torch.matmul(self.K_opt, var.unsqueeze(-1)).squeeze(-1)
    
    def obs(self, var):
        obs = torch.matmul(self.D, var.unsqueeze(-1)).squeeze(-1) # (batchsize, koopman_dim) --> (batchsize, in_channels * input_1d_width)
        obs = obs.view(-1, self.in_channels, self.input_1d_width).unsqueeze(1) # (batchsize, in_channels * input_1d_width) --> (batchsize, 1, in_channels, input_1d_width)
        return obs


def generate_original_data(trace_num, time_step=100, observation_dim=4, koopman_dim=2):
    
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 1)
            m.bias.data.fill_(0.01)
    
    if os.path.exists(f'Data/origin/k_{koopman_dim}/origin.npz'): 
        return
    else:
        os.makedirs(f'Data/origin/k_{koopman_dim}/', exist_ok=True)
    
    # params
    scale = 10
    
    # models
    system = Koopman_System(in_channels=1, input_1d_width=observation_dim, koopman_dim=koopman_dim).apply(init_weights)

    # generate koopman system observation traces
    trace_var = []
    trace_obs = []
    with torch.no_grad():
        for _ in tqdm(range(trace_num)):
            
            var = scale * (2*(torch.rand(1, koopman_dim)-0.5)) # random generate a koopman variable
            koopman_var_series = [var]
            koopman_obs_series = [system.obs(var)] # record its representation in observation space

            for _ in range(1, time_step+1):
                next_var = system.k_evol(var) # evolve in koopman space
                next_obs = system.obs(next_var) # recover to observation space
                koopman_var_series.append(next_var)
                koopman_obs_series.append(next_obs)
                var = next_var
            
            trace_var.append(torch.concat(koopman_var_series, dim=0).squeeze(0).numpy())    
            trace_obs.append(torch.concat(koopman_obs_series, dim=1).squeeze(0).numpy())
    
    # plot koopman evolving trace
    for i, tr in enumerate(trace_obs[:2]):
        plt.figure(figsize=(16,16))
        for j in range(observation_dim):
            plt.subplot(observation_dim,1,j+1)
            plt.plot(tr[:,0,j])
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(f'Data/origin/k_{koopman_dim}/trace_obs_{i+1}.jpg', dpi=300)
        plt.close()
    for i, tr in enumerate(trace_var[:2]):
        plt.figure(figsize=(16,16))
        for j in range(koopman_dim):
            plt.subplot(koopman_dim,1,j+1)
            plt.plot(tr[:,j])
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(f'Data/origin/k_{koopman_dim}/trace_var_{i+1}.jpg', dpi=300)
        plt.close()
    
    # save
    np.savez(f'Data/origin/k_{koopman_dim}/origin.npz', trace_var=trace_var, trace_obs=trace_obs, time_step=time_step)
    print(f'save origin data form seed 1 to {trace_num} at Data/origin/k_{koopman_dim}/')
    
    
def generate_dataset(trace_num, koopman_dim, tau, sample_num=None, is_print=False, sequence_length=None):

    if sequence_length is not None and os.path.exists(f"Data/data/k_{koopman_dim}/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/data/k_{koopman_dim}/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/data/k_{koopman_dim}/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif sequence_length is None and os.path.exists(f"Data/data/k_{koopman_dim}/tau_{tau}/train.npz") and os.path.exists(f"Data/data/k_{koopman_dim}/tau_{tau}/val.npz") and os.path.exists(f"Data/data/k_{koopman_dim}/tau_{tau}/test.npz"):
        return
    
    # load original data
    if is_print: print('loading original trace data:')
    tmp = np.load(f"Data/origin/k_{koopman_dim}/origin.npz")
    data = np.array(tmp['trace_obs'])[:trace_num] # (trace_num, time_length, channel, feature_num)

    # subsampling
    subsampling = tau if tau!=0 else 1
    data = data[:, ::subsampling]
    point_num = int(tmp['time_step'] / subsampling) - 1
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

    # save statistic information
    data_dir = f"Data/data/k_{koopman_dim}/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean.txt", np.mean(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_std.txt", np.std(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_max.txt", np.max(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_min.txt", np.min(data, axis=(0,1)))
    np.savetxt(data_dir + "/tau.txt", [tau]) # Save the timestep

    # single-sample time steps
    if sequence_length is None:
        sequence_length = 2 if tau != 0. else 1
    
    ##################################
    # Create [train,val,test] dataset
    ##################################
    train_num = int(0.8*trace_num)
    val_num = int(0.1*trace_num)
    test_num = int(0.1*trace_num)
    trace_list = {'train':range(train_num), 'val':range(train_num,train_num+val_num), 'test':range(train_num+val_num,train_num+val_num+test_num)}
    for item in ['train','val','test']:
                
        # select trace num
        N_TRACE = len(trace_list[item])
        data_item = data[trace_list[item]]

        # select sliding window index from N trace
        idxs_timestep = []
        idxs_ic = []
        for ic in range(N_TRACE):
            seq_data = data_item[ic]
            idxs = np.arange(0, np.shape(seq_data)[0]-sequence_length, 1)
            for idx_ in idxs:
                idxs_ic.append(ic)
                idxs_timestep.append(idx_)
            if is_print: print(f'\rtau[{tau}] {item} data process_1[{ic+1}/{N_TRACE}]', end='')

        # generator item dataset
        sequences = []
        for bn in range(len(idxs_timestep)):
            idx_ic = idxs_ic[bn]
            idx_timestep = idxs_timestep[bn]
            tmp = data_item[idx_ic, idx_timestep:idx_timestep+sequence_length]
            sequences.append(tmp)
            if is_print: print(f'\rtau[{tau}] {item} data process_2[{bn+1}/{len(idxs_timestep)}]', end='')

        sequences = np.array(sequences) 
        if is_print: print(f'tau[{tau}]', f"{item} dataset", np.shape(sequences))

        # keep sequences_length equal to sample_num
        if sample_num is not None:
            repeat_num = int(np.floor(N_TRACE*sample_num/len(sequences)))
            idx = np.random.choice(range(len(sequences)), N_TRACE*sample_num-len(sequences)*repeat_num, replace=False)
            idx = np.sort(idx)
            tmp1 = sequences[idx]
            tmp2 = None
            for i in range(repeat_num):
                if i == 0:
                    tmp2 = sequences
                else:
                    tmp2 = np.concatenate((tmp2, sequences), axis=0)
            sequences = tmp1 if tmp2 is None else np.concatenate((tmp1, tmp2), axis=0)
        if is_print: print(f'tau[{tau}]', f"after process", np.shape(sequences))

        # save item dataset
        if sequence_length!=1 and sequence_length!=2:
            np.savez(data_dir+f'/{item}_{sequence_length}.npz', data=sequences)
        else:
            np.savez(data_dir+f'/{item}.npz', data=sequences)

        # plot
        if sequence_length==1 or sequence_length==2:
            plt.figure(figsize=(16,10))
            plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
            plt.plot(sequences[:point_num,0,0,0], label='c1')
            plt.plot(sequences[:point_num,0,0,1], label='c2')
            plt.plot(sequences[:point_num,0,0,2], label='c3')
            plt.plot(sequences[:point_num,0,0,3], label='c4')
            plt.legend()
            plt.savefig(data_dir+f'/{item}.jpg', dpi=300)