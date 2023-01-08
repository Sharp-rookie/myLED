# -*- coding: utf-8 -*-
import os
import numpy as np


def generate_dataset(trace_num, tau, sample_num=None, is_print=False, sequence_length=None):
    
    if sequence_length is not None and os.path.exists(f"Data/data/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif sequence_length is None and os.path.exists(f"Data/data/tau_{tau}/train.npz") and os.path.exists(f"Data/data/tau_{tau}/val.npz") and os.path.exists(f"Data/data/tau_{tau}/test.npz"):
        return

    # Load original data
    if is_print: print('loading original trace data ...')
    simdata = np.load("Data/origin/lattice_boltzmann.npz")['data']
    rho_act_all = np.array(simdata["rho_act_all"])[:, np.newaxis]
    rho_in_all = np.array(simdata["rho_in_all"])[:, np.newaxis]
    data = np.concatenate((rho_act_all, rho_in_all), axis=1)
    data = np.transpose(data, (0,2,1,3))
    if is_print: print('original data shape', np.shape(data), '# (trace_num, time_length, feature_num, feature_space_dim)')
    
    # subsampling
    dt = simdata["dt"]
    subsampling = int(tau/dt) if tau!=0. else 1
    data = data[:, ::subsampling]
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, feature_num, feature_space_dim)')
    
    # save statistic information
    data_dir = f"Data/data/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean.txt", np.mean(data, axis=(0,1,3)))
    np.savetxt(data_dir + "/data_std.txt", np.std(data, axis=(0,1,3)))
    np.savetxt(data_dir + "/data_max.txt", np.max(data, axis=(0,1,3)))
    np.savetxt(data_dir + "/data_min.txt", np.min(data, axis=(0,1,3)))
    np.savetxt(data_dir + "/tau.txt", [tau]) # Save the timestep
    
    # single-sample time steps for train
    if sequence_length is None:
        sequence_length = 2 if tau != 0. else 1
    
    #######################j
    # Create [train,val,test] dataset
    #######################
    trace_list = {'train':range(3), 'val':range(3,4), 'test':range(4,5)}
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
        if is_print: print(f'\ntau[{tau}]', f"{item} dataset", np.shape(sequences))
        
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
            # TODO: 画热度图
            pass

generate_dataset(1,0.5,1,1,1);exit(0)