import os
import numpy as np
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
import multiprocessing
from multiprocessing import Process
import warnings;warnings.simplefilter('ignore')

from .gillespie import generate_toggle_switch_origin


def findNearestPoint(data_t, start=0, object_t=10.0):
    """Find the nearest time point to object time"""

    index = start

    if index >= len(data_t):
        return index

    while not (data_t[index] <= object_t and data_t[index+1] > object_t):
        if index < len(data_t)-2:
            index += 1
        elif index == len(data_t)-2: # last one
            index += 1
            break
    
    return index


def time_discretization(seed, total_t, dt=None, is_print=False, data=None, save=True):
    """Time-forward NearestNeighbor interpolate to discretizate the time"""

    if not data:
        data = np.load(f'Data/ToggleSwitch/origin/{seed}/origin.npz')
    
    data_t = data['t']
    data_Gx = data['Gx']
    data__Gx = data['_Gx']
    data_Px = data['Px']
    data_Gy = data['Gy']
    data__Gy = data['_Gy']
    data_Py = data['Py']

    dt = 5e-6 if dt is None else dt
    current_t = 0.0
    index = 0
    t, Gx, _Gx, Px, Gy, _Gy, Py = [], [], [], [], [], [], []
    while current_t < total_t:
        index = findNearestPoint(data_t, start=index, object_t=current_t)
        t.append(current_t)
        Gx.append(data_Gx[index])
        _Gx.append(data__Gx[index])
        Px.append(data_Px[index])
        Gy.append(data_Gy[index])
        _Gy.append(data__Gy[index])
        Py.append(data_Py[index])

        current_t += dt

        if is_print == 1: print(f'\rSeed[{seed}] interpolating {current_t:.6f}/{total_t}', end='')

    if save:
        np.savez(f'Data/ToggleSwitch/origin/{seed}/data.npz', dt=dt, t=t, Gx=Gx, _Gx=_Gx, Px=Px, Gy=Gy, _Gy=_Gy, Py=Py)
        
        plt.figure(figsize=(16,9))
        name = ['G_x', 'G_x2', 'P_x', 'G_y', 'G_y2', 'P_y']
        data = [Gx, _Gx, Px, Gy, _Gy, Py]
        for i in range(6):
            ax = plt.subplot(2,3,1+i)
            ax.plot(t, data[i])
            ax.set_xlabel(r'$t / s$', fontsize=18)
            ax.set_ylabel(rf'${name[i]}$', fontsize=18)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.9,
            bottom=0.15,
            wspace=0.2
        )
        plt.savefig(f'Data/ToggleSwitch/origin/{seed}/data.pdf', dpi=300)
    else:
        return np.array([Px, Py, Gx, _Gx, Gy, _Gy])


def generate_original_data(trace_num, total_t, dt, save=True, plot=False, parallel=False):

    os.makedirs('Data/ToggleSwitch/origin', exist_ok=True)

    # generate original data by gillespie algorithm
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/ToggleSwitch/origin/{seed}/origin.npz'):
            IC = [1,0,0,1,0,0]
            if parallel:
                is_print = len(subprocess)==0
                subprocess.append(Process(target=generate_toggle_switch_origin, args=(total_t, seed, IC, True, is_print), daemon=True))
                subprocess[-1].start()
            else:
                generate_toggle_switch_origin(total_t, seed, IC, is_print=True)
    while any([subp.exitcode == None for subp in subprocess]):
        pass
    
    # time discretization by time-forward NearestNeighbor interpolate
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/ToggleSwitch/origin/{seed}/data.npz'):
            is_print = len(subprocess)==0
            if parallel:
                subprocess.append(Process(target=time_discretization, args=(seed, total_t, dt, is_print), daemon=True))
                subprocess[-1].start()
            else:
                time_discretization(seed, total_t, dt, is_print)
    while any([subp.exitcode == None for subp in subprocess]):
        pass

    print(f'save origin data form seed 1 to {trace_num} at Data/ToggleSwitch/origin/')


def generate_original_data2(trace_num, total_t, dt, parallel=False):

    def unit(queue, total_t, seed, dt, IC, is_print):
        simdata = generate_toggle_switch_origin(total_t=total_t, seed=seed, IC=IC, save=False, is_print=is_print)
        data = time_discretization(seed=seed, total_t=total_t, dt=dt, is_print=False, data=simdata, save=False)
        queue.put_nowait(data)

    parallel_batch = 90 # max parallel subprocess num at same time

    queue = multiprocessing.Manager().Queue()
    
    sol = []
    for batch_id in range(int(trace_num/parallel_batch)+1):
        print(f'[{batch_id+1}/{int(trace_num/parallel_batch)+1}] parallel batch')
        subprocess = []

        for i in range(batch_id*parallel_batch+1, (batch_id+1)*parallel_batch+1):
            if i > trace_num: break
            IC = [1,0,0,1,0,0]
            if parallel:
                is_print = len(subprocess)==0
                subprocess.append(Process(target=unit, args=(queue, total_t, i, dt, IC, is_print)))
                subprocess[-1].start()
            else: 
                sol.append(unit(queue, total_t, i, dt, IC, True))
        
        while any([subp.exitcode == None for subp in subprocess]):
            if not queue.empty():
                data = queue.get_nowait()
                sol.append(data)
    
    return np.array(sol).transpose((0,2,1))


def generate_dataset_static(trace_num, tau=0., dt=0.01, max_tau=5., is_print=False, parallel=False):
    
    if os.path.exists(f"Data/ToggleSwitch/data/tau_{tau}/train_static.npz") and os.path.exists(f"Data/ToggleSwitch/data/tau_{tau}/val_static.npz") and os.path.exists(f"Data/ToggleSwitch/data/tau_{tau}/test_static.npz"):
        return
    
    # generate simulation data
    os.makedirs('Data/ToggleSwitch/data/', exist_ok=True)
    if not os.path.exists(f"Data/ToggleSwitch/data/static_{max_tau:.2f}.npz"):
        if is_print: print('generating simulation trajectories:')
        data = generate_original_data2(trace_num, total_t=max_tau+dt, dt=dt, parallel=parallel)
        data = data[:,:,np.newaxis] # add channel dim
        np.savez(f"Data/ToggleSwitch/data/static_{max_tau:.2f}.npz", data=data)
        if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')
    else:
        data = np.load(f"Data/ToggleSwitch/data/static_{max_tau:.2f}.npz")['data']

    if is_print: print(f"\r[{tau}/{max_tau}]", end='')

    # save statistic information
    data_dir = f"Data/ToggleSwitch/data/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean_static.txt", np.mean(data, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/data_std_static.txt", np.std(data, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/data_max_static.txt", np.max(data, axis=(0,1)).reshape(1,-1))
    np.savetxt(data_dir + "/data_min_static.txt", np.min(data, axis=(0,1)).reshape(1,-1))
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
        squence_length = 2 if tau!=0. else 1

        assert step_length<=data_item.shape[1], f"Tau {tau} is larger than the simulation time length {dt*data_item.shape[1]}"
        sequences = data_item[:, ::step_length]
        sequences = sequences[:, :squence_length]
        
        # save
        np.savez(data_dir+f'/{item}_static.npz', data=sequences)

        # # plot
        # name = ['P_x', 'P_y', 'G_x', 'G_x2', 'G_y', 'G_y2']

        # plt.figure(figsize=(16,10))
        # plt.title(f'{item.capitalize()} Data')
        # for i in range(6):
        #     ax = plt.subplot(2,3,1+i)
        #     ax.plot(sequences[:,0,0,i])
        #     ax.set_xlabel(r'$t / s$', fontsize=18)
        #     ax.set_ylabel(rf'${name[i]}$', fontsize=18)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.2)
        # plt.savefig(data_dir+f'/{item}_static_input.pdf', dpi=300)

        # plt.figure(figsize=(16,10))
        # plt.title(f'{item.capitalize()} Data')
        # for i in range(6):
        #     ax = plt.subplot(2,3,1+i)
        #     ax.plot(sequences[:,squence_length-1,0,i])
        #     ax.set_xlabel(r'$t / s$', fontsize=18)
        #     ax.set_ylabel(rf'${name[i]}$', fontsize=18)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.2)
        # plt.savefig(data_dir+f'/{item}_static_target.pdf', dpi=300)
    
    
def generate_dataset_slidingwindow(trace_num, tau, sample_num=None, is_print=False, sequence_length=None):

    if (sequence_length is not None) and os.path.exists(f"Data/ToggleSwitch/data/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/ToggleSwitch/data/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/ToggleSwitch/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif (sequence_length is None) and os.path.exists(f"Data/ToggleSwitch/data/tau_{tau}/train.npz") and os.path.exists(f"Data/ToggleSwitch/data/tau_{tau}/val.npz") and os.path.exists(f"Data/ToggleSwitch/data/tau_{tau}/test.npz"):
        return

    # load original data
    if is_print: print('loading original trace data:')
    data = []
    iter = tqdm(range(1, trace_num+1)) if is_print else range(1, trace_num+1)
    for trace_id in iter:
        tmp = np.load(f"Data/ToggleSwitch/origin/{trace_id}/data.npz")
        dt = tmp['dt']
        Gx = np.array(tmp['Gx'])[:, np.newaxis, np.newaxis] # (sample_num, channel, feature_num)
        _Gx = np.array(tmp['_Gx'])[:, np.newaxis, np.newaxis]
        Px = np.array(tmp['Px'])[:, np.newaxis, np.newaxis]
        Gy = np.array(tmp['Gy'])[:, np.newaxis, np.newaxis]
        _Gy = np.array(tmp['_Gy'])[:, np.newaxis, np.newaxis]
        Py = np.array(tmp['Py'])[:, np.newaxis, np.newaxis]

        trace = np.concatenate((Px, Py, Gx, _Gx, Gy, _Gy), axis=-1)
        data.append(trace[np.newaxis])
    data = np.concatenate(data, axis=0)

    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

    # save statistic information
    data_dir = f"Data/ToggleSwitch/data/tau_{tau}"
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
        if is_print: print(f'tau[{tau}]', f"{item} dataset (sequence_length={sequence_length})", np.shape(sequences))

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
            name = ['P_x', 'P_y', 'G_x', 'G_x2', 'G_y', 'G_y2']
            plt.figure(figsize=(16,10))
            plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
            for i in range(6):
                ax = plt.subplot(2,3,i+1)
                ax.set_title(rf'{name[i]}')
                plt.plot(sequences[:, 0, 0, i])
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.2)
            plt.savefig(data_dir+f'/{item}_input.pdf', dpi=300)

            plt.figure(figsize=(16,10))
            plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
            for i in range(6):
                ax = plt.subplot(2,3,i+1)
                ax.set_title(rf'{name[i]}')
                plt.plot(sequences[:, sequence_length-1, 0, i])
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.2)
            plt.savefig(data_dir+f'/{item}_target.pdf', dpi=300)