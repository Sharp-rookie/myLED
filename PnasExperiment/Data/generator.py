import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
import warnings;warnings.simplefilter('ignore')

from .gillespie import generate_origin


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


def time_discretization(seed, total_t, origin_dt=False):
    """Time-forward NearestNeighbor interpolate to discretizate the time"""

    data = np.load(f'Data/origin/{seed}/origin.npz')
    data_t = data['t']
    data_X = data['X']
    data_Y = data['Y']
    data_Z = data['Z']

    dt = 5e-6 if origin_dt else 5e-3 # 5e-6是手动pnas这个实验仿真得出的时间间隔大概平均值
    current_t = 0.0
    index = 0
    t, X, Y, Z = [], [], [], []
    while current_t < total_t:
        index = findNearestPoint(data_t, start=index, object_t=current_t)
        t.append(current_t)
        X.append(data_X[index])
        Y.append(data_Y[index])
        Z.append(data_Z[index])

        current_t += dt

        if seed == 1:
            print(f'\rSeed[{seed}] interpolating {current_t:.6f}/{total_t}', end='')

    plt.figure(figsize=(16,4))
    plt.title(f'dt = {dt}')
    ax1 = plt.subplot(1,3,1)
    ax1.set_title('X')
    plt.plot(t, X, label='X')
    plt.xlabel('time / s')
    ax2 = plt.subplot(1,3,2)
    ax2.set_title('Y')
    plt.plot(t, Y, label='Y')
    plt.xlabel('time / s')
    ax3 = plt.subplot(1,3,3)
    ax3.set_title('Z')
    plt.plot(t, Z, label='Z')
    plt.xlabel('time / s')

    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.9,
        bottom=0.1,
        wspace=0.2
    )
    plt.savefig(f'Data/origin/{seed}/data.png', dpi=500)

    np.savez(f'Data/origin/{seed}/data.npz', dt=dt, t=t, X=X, Y=Y, Z=Z)


def generate_original_data(trace_num, total_t):

    os.makedirs('Data/origin', exist_ok=True)

    # generate original data by gillespie algorithm
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/origin/{seed}/origin.npz'):
            IC = [np.random.randint(5,200), np.random.randint(5,100), np.random.randint(0,5000)]
            subprocess.append(Process(target=generate_origin, args=(total_t, seed, IC), daemon=True))
            subprocess[-1].start()
            print(f'\rStart process[seed={seed}] for origin data' + ' '*30)
        else:
            pass
    while any([subp.exitcode == None for subp in subprocess]):
        pass
    
    # time discretization by time-forward NearestNeighbor interpolate
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/origin/{seed}/data.npz'):
            subprocess.append(Process(target=time_discretization, args=(seed, total_t, True), daemon=True))
            subprocess[-1].start()
            print(f'\rStart process[seed={seed}] for time-discrete data' + ' '*30)
    while any([subp.exitcode == None for subp in subprocess]):
        pass

    print(f'save origin data form seed 1 to {trace_num} at Data/origin/')
    
    
def generate_dataset(trace_num, tau, sample_num=None, is_print=False):

    if os.path.exists(f"Data/data/tau_{tau}/train.npz") and os.path.exists(f"Data/data/tau_{tau}/val.npz") and os.path.exists(f"Data/data/tau_{tau}/test.npz"):
        return

    # load original data
    if is_print: print('loading original trace data:')
    data = []
    from tqdm import tqdm
    for trace_id in tqdm(range(1, trace_num+1)):
        tmp = np.load(f"Data/origin/{trace_id}/data.npz")
        X = np.array(tmp['X'])[:, np.newaxis]
        Y = np.array(tmp['Y'])[:, np.newaxis]
        Z = np.array(tmp['Z'])[:, np.newaxis]

        trace = np.concatenate((X, Y, Z), axis=1)
        data.append(trace[np.newaxis])
    data = np.concatenate(data, axis=0)

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

    # single-sample time steps for train
    sequence_length = 2 if tau != 0. else 1

    #######################j
    # Create [train,val,test] dataset
    #######################
    trace_list = {'train':range(256), 'val':range(256,288), 'test':range(288,320)}
    for item in ['train','val','test']:
        
        if os.path.exists(data_dir+f'/{item}.npz'): continue
        
        # select trace num
        N_TRACE = len(trace_list[item])
        data_item = data[trace_list[item]]

        # select sliding window index from 2 trace
        idxs_timestep = []
        idxs_ic = []
        for ic in range(N_TRACE):
            seq_data = data_item[ic]
            idxs = np.arange(0, np.shape(seq_data)[0]-sequence_length, 1)
            for idx_ in idxs:
                idxs_ic.append(ic)
                idxs_timestep.append(idx_)

        # generator item dataset
        sequences = []
        for bn in range(len(idxs_timestep)):
            idx_ic = idxs_ic[bn]
            idx_timestep = idxs_timestep[bn]
            tmp = data_item[idx_ic, idx_timestep:idx_timestep+sequence_length]
            sequences.append(tmp)

        sequences = np.array(sequences) 
        if is_print: print(f'tau[{tau}]', f"original {item} dataset", np.shape(sequences))

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
        if is_print: print(f'tau[{tau}]', f"processed {item} dataset", np.shape(sequences))

        # save item dataset
        np.savez(data_dir+f'/{item}.npz', data=sequences)

        # plot
        plt.figure(figsize=(16,10))
        plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
        for i in range(3):
            ax = plt.subplot(3,1,i+1)
            ax.set_title(['X','Y','Z'][i])
            plt.plot(sequences[:, 0, i])
        plt.subplots_adjust(left=0.05, bottom=0.05,  right=0.95,  top=0.95,  hspace=0.35)
        plt.savefig(data_dir+f'/{item}.jpg', dpi=300)


if __name__ == '__main__':

    # generate original data
    trace_num = 256+32+32
    total_t = 9
    generate_original_data(trace_num=trace_num, total_t=total_t)