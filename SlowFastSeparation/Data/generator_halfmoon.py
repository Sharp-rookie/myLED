import os
import numpy as np
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
from sdeint import itoSRI2, itoEuler
import warnings;warnings.simplefilter('ignore')

from util import seed_everything


class SDE_HalfMoon():
    def __init__(self, a1, a2, a3, a4):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
    
    def f(self, x, t):
        return np.array([self.a1 * 1, 
                         self.a3 * (1-x[1])])
    
    def g(self, x, t):
        return np.array([[self.a2*1., 0.*0.], 
                        [0.*0., self.a4*1.]])


def generate_original_data(trace_num, total_t=6280, dt=0.1, save=True, plot=False, parallel=False):
    
    def solve_1_trace(trace_id=0, total_t=5, dt=0.001):
        
        seed_everything(trace_id)
        
        sde = SDE_HalfMoon(a1 = 1e-3, a2 = 1e-3, a3 = 1e-1, a4 = 1e-1)
        y0 = [1., 0.]
        tspan  =np.arange(0, total_t, dt)
        
        sol = itoSRI2(sde.f, sde.g, y0, tspan) # Runge-Kutta algorithm
        # sol = itoEuler(sde.f, sde.g, y0, tspan) # Euler-Maruyama algorithm

        x = sol[:, 1] * np.cos(sol[:, 0]+sol[:, 1]-1)
        y = sol[:, 1] * np.sin(sol[:, 0]+sol[:, 1]-1)
        result = np.concatenate((x[...,np.newaxis],y[...,np.newaxis],sol), axis=1)

        if plot:
            os.makedirs('Data/HalfMoon/origin', exist_ok=True)
            plt.figure(figsize=(16, 16))
            ax1 = plt.subplot(2,2,1)
            ax1.plot(tspan, result[:, 2], label='u')
            ax1.plot(tspan, result[:, 3], label='v')
            ax1.set_xlabel('t')
            ax1.set_ylabel('var')
            ax1.legend()
            ax2 = plt.subplot(2,2,2)
            ax2.scatter(result[::5,0], result[::5,1], c=tspan[::5], cmap='viridis', linewidths=0.01)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax3 = plt.subplot(2,2,3)
            ax3.plot(tspan, result[:,0], label='x')
            ax3.plot(tspan, result[:,1], label='y')
            ax3.set_xlabel('t')
            ax3.set_ylabel('var')
            ax1.legend()
            plt.savefig(f'Data/HalfMoon/origin/halfmoon_{trace_id}.pdf', dpi=100)
        
        return np.array(result)
    
    if save and os.path.exists('Data/HalfMoon/origin/origin.npz'): return
    
    trace = []
    for trace_id in tqdm(range(1, trace_num+1)):
        sol = solve_1_trace(trace_id, total_t, dt)
        trace.append(sol)
    
    if save: 
        os.makedirs('Data/HalfMoon/origin', exist_ok=True)
        np.savez('Data/HalfMoon/origin/origin.npz', trace=trace, dt=dt, T=total_t)
        print(f'save origin data form seed 1 to {trace_num} at Data/HalfMoon/origin/')
    
    return np.array(trace)
# generate_original_data(1, total_t=31400, dt=1, save=False, plot=True)


def generate_dataset_static(trace_num, tau=0., dt=0.01, max_tau=5., is_print=False, parallel=False):

    if os.path.exists(f"Data/HalfMoon/data/tau_{tau}/train_static.npz") and os.path.exists(f"Data/HalfMoon/data/tau_{tau}/val_static.npz") and os.path.exists(f"Data/HalfMoon/data/tau_{tau}/test_static.npz"):
        return
    
    # generate simulation data
    if not os.path.exists(f"Data/HalfMoon/data/static_{max_tau:.2f}.npz"):
        if is_print: print('generating simulation trajectories:')
        data = generate_original_data(trace_num, total_t=max_tau+101*dt, dt=dt, save=False, plot=False, parallel=parallel)
        data = data[:,:,np.newaxis] # add channel dim
        np.savez(f"Data/HalfMoon/data/static_{max_tau:.2f}.npz", data=data)
        if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')
    else:
        data = np.load(f"Data/HalfMoon/data/static_{max_tau:.2f}.npz")['data']

    if is_print: print(f"\r[{tau}/{max_tau}]", end='')

    # save statistic information
    data_dir = f"Data/HalfMoon/data/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean_static.txt", np.mean(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_std_static.txt", np.std(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_max_static.txt", np.max(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_min_static.txt", np.min(data, axis=(0,1)))
    np.savetxt(data_dir + "/tau_static.txt", [tau]) # Save the timestep

    ##################################
    # Create [train,val,test] dataset
    ##################################
    train_num = int(0.7*trace_num)
    val_num = int(0.1*trace_num)
    test_num = int(0.2*trace_num)
    trace_list = {'train':range(train_num), 'val':range(train_num,train_num+val_num), 'test':range(train_num+val_num,train_num+val_num+test_num)}
    for item in ['train','val','test']:
                
        # select trace num
        data_item = data[trace_list[item]]

        # subsampling
        step_length = int(tau/dt) if tau!=0. else 1

        assert step_length<data_item.shape[1], f"Tau {tau} is larger than the simulation time length{dt*data_item.shape[1]}"
        sequences = data_item[:, 100::step_length]
        sequences = sequences[:, :2]
        
        # save
        np.savez(data_dir+f'/{item}_static.npz', data=sequences)

        # plot
        plt.figure(figsize=(16,10))
        plt.title(f'{item.capitalize()} Data')
        plt.plot(sequences[:,0,0,0], label='x')
        plt.plot(sequences[:,0,0,1], label='y')
        plt.legend()
        plt.savefig(data_dir+f'/{item}_static_input.pdf', dpi=300)

        plt.figure(figsize=(16,10))
        plt.title(f'{item.capitalize()} Data')
        plt.plot(sequences[:,1,0,0], label='x')
        plt.plot(sequences[:,1,0,1], label='y')
        plt.legend()
        plt.savefig(data_dir+f'/{item}_static_target.pdf', dpi=300)

    
def generate_dataset_slidingwindow(trace_num, tau, sample_num=None, is_print=False, sequence_length=None):

    if (sequence_length is not None) and os.path.exists(f"Data/HalfMoon/data/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/HalfMoon/data/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/HalfMoon/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif (sequence_length is None) and os.path.exists(f"Data/HalfMoon/data/tau_{tau}/train.npz") and os.path.exists(f"Data/HalfMoon/data/tau_{tau}/val.npz") and os.path.exists(f"Data/HalfMoon/data/tau_{tau}/test.npz"):
        return
    
    # load original data
    if is_print: print('loading original trace data:')
    tmp = np.load(f"Data/HalfMoon/origin/origin.npz")
    dt = tmp['dt']
    data = np.array(tmp['trace'])[:trace_num,:,np.newaxis] # (trace_num, time_length, channel, feature_num)
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

    # save statistic information
    data_dir = f"Data/HalfMoon/data/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean.txt", np.mean(data, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/data_std.txt", np.std(data, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/data_max.txt", np.max(data, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/data_min.txt", np.min(data, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/tau.txt", [tau]) # Save the timestep

    # single-sample time steps
    if sequence_length is None:
        sequence_length = 2 if tau != 0. else 1
        seq_none = True
    else:
        seq_none = False
    
    ##################################
    # Create [train,val,test] dataset
    ##################################
    train_num = int(0.7*trace_num)
    val_num = int(0.1*trace_num)
    test_num = int(0.2*trace_num)
    trace_list = {'train':range(train_num), 'val':range(train_num,train_num+val_num), 'test':range(train_num+val_num,train_num+val_num+test_num)}
    for item in ['train','val','test']:
                
        # select trace num
        N_TRACE = len(trace_list[item])
        data_item = data[trace_list[item]]

        # subsampling
        step_length = int(tau/dt) if tau!=0. else 1

        # select sliding window index from N trace
        idxs_timestep = []
        idxs_ic = []
        for ic in range(N_TRACE):
            seq_data = data_item[ic]
            idxs = np.arange(0, np.shape(seq_data)[0]-step_length*(sequence_length-1), 1)
            for idx_ in idxs:
                idxs_ic.append(ic)
                idxs_timestep.append(idx_)

        # generator item dataset
        sequences = []
        parallel_sequences = [[] for _ in range(N_TRACE)]
        for bn in range(len(idxs_timestep)):
            idx_ic = idxs_ic[bn]
            idx_timestep = idxs_timestep[bn]
            tmp = data_item[idx_ic, idx_timestep : idx_timestep+step_length*(sequence_length-1)+1 : step_length]
            sequences.append(tmp)
            parallel_sequences[idx_ic].append(tmp)
            if is_print: print(f'\rtau[{tau}] sliding window for {item} data [{bn+1}/{len(idxs_timestep)}]', end='')
        if is_print: print()

        sequences = np.array(sequences) 
        if is_print: print(f'tau[{tau}]', f"{item} dataset (sequence_length={sequence_length}, step_length={step_length})", np.shape(sequences))

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

        # save
        if not seq_none:
            np.savez(data_dir+f'/{item}_{sequence_length}.npz', data=sequences)
        else:
            np.savez(data_dir+f'/{item}.npz', data=sequences)

            # plot
            if seq_none:
                plt.figure(figsize=(16,10))
                plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
                plt.plot(sequences[:,0,0,0], label='x')
                plt.plot(sequences[:,0,0,1], label='y')
                plt.legend()
                plt.savefig(data_dir+f'/{item}_input.pdf', dpi=300)

                plt.figure(figsize=(16,10))
                plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
                plt.plot(sequences[:,sequence_length-1,0,0], label='x')
                plt.plot(sequences[:,sequence_length-1,0,1], label='y')
                plt.legend()
                plt.savefig(data_dir+f'/{item}_target.pdf', dpi=300)
