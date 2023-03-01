import os
import numpy as np
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
from scipy.integrate import odeint
import warnings;warnings.simplefilter('ignore')

from util import seed_everything


class FitzHugh_Nagumo():
    def __init__(self, a, c, I, g):
        self.a = a
        self.c = c
        self.I = I
        self.g = g

    def __call__(self, y0, t):
        
        v, w, b = y0

        dv = self.a * (-v * (v-1) * (v-b) - w + self.I)
        dw = v - self.c * w
        db = self.g(t)

        return [dv, dw, db]


def generate_original_data(trace_num, total_t=20.1, dt=0.01, save=True, plot=False, parallel=False):
    
    def solve_1_trace(trace_id=0, total_t=5, dt=0.01):
        
        seed_everything(trace_id)
        
        y0 = [np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(0,1)]

        t  =np.arange(0, total_t, dt)
        period = 12
        a, c, I = 1e7, 1., 0.2
        g = lambda x: np.pi/period * np.sin(x*2*np.pi/period)
        sol = odeint(FitzHugh_Nagumo(a,c,I,g), y0, t)

        if plot:
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            ax1.plot(t, sol[:,0], color='blue', label='v')
            ax1.plot(t, sol[:,1], color='green', label='w')
            ax2.plot(t, sol[:,2], color='red')
            ax1.legend()
            ax2.set_ylabel('b')
            ax1.set_position([0.1, 0.35, 0.8, 0.6])
            ax2.set_position([0.1, 0.1, 0.8, 0.2])
            plt.savefig(f'Data/FHN/origin/fhn_{trace_id}.pdf', dpi=300)
        
        return sol
    
    if save and os.path.exists('Data/FHN/origin/origin.npz'): return
    
    trace = []
    for trace_id in tqdm(range(1, trace_num+1)):
        sol = solve_1_trace(trace_id, total_t, dt)
        trace.append(sol)
    
    if save: 
        os.makedirs('Data/FHN/origin', exist_ok=True)
        np.savez('Data/FHN/origin/origin.npz', trace=trace, dt=dt, T=total_t)

    print(f'save origin data form seed 1 to {trace_num} at Data/origin/')

    return trace

    
def generate_dataset(trace_num, tau, sample_num=None, is_print=False, sequence_length=None):

    if (sequence_length is not None) and os.path.exists(f"Data/FHN/data/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/FHN/data/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/FHN/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif (sequence_length is None) and os.path.exists(f"Data/FHN/data/tau_{tau}/train.npz") and os.path.exists(f"Data/FHN/data/tau_{tau}/val.npz") and os.path.exists(f"Data/FHN/data/tau_{tau}/test.npz"):
        return
    
    # load original data
    if is_print: print('loading original trace data:')
    tmp = np.load(f"Data/FHN/origin/origin.npz")
    dt = tmp['dt']
    data = np.array(tmp['trace'])[:trace_num,:,np.newaxis] # (trace_num, time_length, channel, feature_num)
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

    # save statistic information
    data_dir = f"Data/FHN/data/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean.txt", np.mean(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_std.txt", np.std(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_max.txt", np.max(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_min.txt", np.min(data, axis=(0,1)))
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
                plt.plot(sequences[:,0,0,0], label='v')
                plt.plot(sequences[:,0,0,1], label='w')
                plt.plot(sequences[:,0,0,2], label='b')
                plt.legend()
                plt.savefig(data_dir+f'/{item}_input.pdf', dpi=300)

                plt.figure(figsize=(16,10))
                plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
                plt.plot(sequences[:,sequence_length-1,0,0], label='v')
                plt.plot(sequences[:,sequence_length-1,0,1], label='w')
                plt.plot(sequences[:,sequence_length-1,0,2], label='b')
                plt.legend()
                plt.savefig(data_dir+f'/{item}_target.pdf', dpi=300)
