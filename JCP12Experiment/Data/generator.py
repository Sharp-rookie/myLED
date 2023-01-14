import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pytorch_lightning import seed_everything
import warnings;warnings.simplefilter('ignore')


def system_4d(y0, t, para=(0.025,3)):
    epsilon, omega =  para
    c1, c2, c3, c4 = y0
    dc1 = -c1
    dc2 = -2 * c2
    dc3 = -(c3-np.sin(omega*c1)*np.sin(omega*c2))/epsilon - c1*omega*np.cos(omega*c1)*np.sin(omega*c2) - c2*omega*np.cos(omega*c2)*np.sin(omega*c1)
    dc4 = -(c4-1/((1+np.exp(-omega*c1))*(1+np.exp(-omega*c2))))/epsilon - c1*omega*np.exp(-omega*c1)/((1+np.exp(-omega*c2))*((1+np.exp(-omega*c1))**2)) - c2*omega*np.exp(-omega*c2)/((1+np.exp(-omega*c1))*((1+np.exp(-omega*c2))**2))
    return [dc1, dc2, dc3, dc4]


def generate_original_data(trace_num, total_t=5, dt=0.001):
    
    def solve_1_trace(trace_id=0, total_t=5, dt=0.001):
        
        seed_everything(trace_id)
        
        y0 = [np.random.uniform(-10,10) for _ in range(4)]
        t  =np.arange(0, total_t, dt)
        sol = odeint(system_4d, y0, t)

        # plt.figure()
        # plt.plot(t, sol[:,0], label='c1')
        # plt.plot(t, sol[:,1], label='c2')
        # plt.plot(t, sol[:,2], label='c3')
        # plt.plot(t, sol[:,3], label='c4')
        # plt.legend()
        # plt.savefig(f'Data/origin/jcp12_{trace_id}.jpg', dpi=300)
        
        return sol
        
    os.makedirs('Data/origin', exist_ok=True)
    
    trace = []
    for trace_id in tqdm(range(1, trace_num+1)):
        sol = solve_1_trace(trace_id, total_t, dt)
        trace.append(sol)
    
    np.savez('Data/origin/origin.npz', trace=trace, dt=dt, total_t=total_t)

    print(f'save origin data form seed 1 to {trace_num} at Data/origin/')
    
    
def generate_dataset(trace_num, tau, sample_num=None, is_print=False, sequence_length=None):

    if sequence_length is not None and os.path.exists(f"Data/data/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif sequence_length is None and os.path.exists(f"Data/data/tau_{tau}/train.npz") and os.path.exists(f"Data/data/tau_{tau}/val.npz") and os.path.exists(f"Data/data/tau_{tau}/test.npz"):
        return

    # load original data
    if is_print: print('loading original trace data:')
    tmp = np.load(f"Data/origin/origin.npz")
    data = np.array(tmp['trace']) # (trace_num, time_length, feature_num)

    # subsampling
    dt = tmp['dt']
    subsampling = int(tau/dt) if tau!=0. else 1
    data = data[:, ::subsampling]
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, feature_num)')

    # save statistic information
    data_dir = f"Data/data/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean.txt", np.mean(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_std.txt", np.std(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_max.txt", np.max(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_min.txt", np.min(data, axis=(0,1)))
    np.savetxt(data_dir + "/tau.txt", [tau]) # Save the timestep

    # single-sample time steps
    if sequence_length is None:
        sequence_length = 2 if tau != 0. else 1

    #######################
    # Create [train,val,test] dataset
    #######################
    train_num = int(0.8*trace_num)
    val_num = int(0.1*trace_num)
    test_num = int(0.1*trace_num)
    trace_list = {'train':range(train_num), 'val':range(train_num,train_num+val_num), 'test':range(train_num+val_num,train_num+val_num+test_num)}
    for item in ['train','val','test']:
        
        # if os.path.exists(data_dir+f'/{item}.npz'): continue
        
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
            plt.plot(sequences[:,0,0], label='c1')
            plt.plot(sequences[:,0,1], label='c2')
            plt.plot(sequences[:,0,2], label='c3')
            plt.plot(sequences[:,0,3], label='c4')
            plt.legend()
            plt.savefig(data_dir+f'/{item}.jpg', dpi=300)