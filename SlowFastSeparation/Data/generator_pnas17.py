import os
import numpy as np
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
from sdeint import itoSRI2, itoEuler
import warnings;warnings.simplefilter('ignore')

from util import seed_everything


class SDE_FHN():
    def __init__(self, a, epsilon, delta1, delta2, du, xdim):
        self.a = a
        self.epsilon = epsilon
        self.delta1 = delta1
        self.delta2 = delta2
        self.du = du
        self.xdim = xdim
    
    def f(self, x, t):

        # 1-dim
        if self.xdim == 1:
            y = [(x[0] - 1/3*x[0]**3 - x[1])/self.epsilon, x[0] + self.a]
        
        # high-dim
        else:
            y = []
            # u
            for i in range(self.xdim):
                if i == 0:
                    y.append((x[i] - 1/3*x[i]**3 - x[i+self.xdim] + self.du*(x[i+1]+x[i+self.xdim-1]-2*x[i]))/self.epsilon)
                elif i == self.xdim-1:
                    y.append((x[i] - 1/3*x[i]**3 - x[i+self.xdim] + self.du*(x[i-self.xdim+1]+x[i-1]-2*x[i]))/self.epsilon)
                else:
                    y.append((x[i] - 1/3*x[i]**3 - x[i+self.xdim] + self.du*(x[i+1]+x[i-1]-2*x[i]))/self.epsilon)
            # v
            for i in range(self.xdim):
                y.append(x[i] + self.a)

        return np.array(y)
    
    def g(self, x, t):
        return np.diag([self.delta1*1.]*self.xdim + [self.delta2*1.]*self.xdim)


def sample_gaussian_field(n, mu=0.0, sigma=1.0, l=1.0):

    def gaussian_kernel(x1, x2, sigma=1.0, l=1.0):
        return sigma**2 * np.exp(-0.5 * (x1 - x2)**2 / l**2)
        # return np.exp(np.sqrt((x1 - x2)**2) / l)

    x = np.linspace(0, 200, n)

    # covariant matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i,j] = gaussian_kernel(x[i], x[j], sigma=sigma, l=l)
            K[j,i] = K[i,j]

    # Cholesky decomposition
    L = np.linalg.cholesky(K)

    # sample
    u = np.random.normal(size=n)
    f = mu + np.dot(L, u).reshape(n)

    return f


def generate_original_data(trace_num, total_t=6280, dt=0.001, save=True, plot=False, parallel=False, xdim=1, delta=0., du=0.5, data_dir='Data/PNAS17_xdim1/origin'):
    
    def solve_1_trace(trace_id):
        
        seed_everything(trace_id)
        
        a, epsilon, delta1, delta2 = 1.05, 0.01, delta, delta
        sde = SDE_FHN(a=a, epsilon=epsilon, delta1=delta1, delta2=delta2, du=du, xdim=xdim)
        
        # initial condition
        # x0 = [np.random.normal(-3, 0.3) for _ in range(2*xdim)]  # local
        v0 = [np.random.uniform(-3, -2) if np.random.randint(0, 2)==0 else np.random.uniform(2, 3) for _ in range(xdim)]
        u0 = [np.random.uniform(-3, -2) if np.random.randint(0, 2)==0 else np.random.uniform(2, 3) for _ in range(xdim)]
        # f = sample_gaussian_field(xdim, mu=-3.0, sigma=1.0, l=5.0) if np.random.randint(0, 2)==0 else sample_gaussian_field(xdim, mu=3.0, sigma=1.0, l=5.0)
        # u0 = []
        # v0 = f.tolist()
        # for v in f:
        #     func = np.poly1d([-1/3,0,-1,-v])
        #     for root in func.roots:
        #         if np.imag(root) == 0:
        #             u0.append(np.real(root))
        #             continue
        x0 = u0 + v0

        tspan  =np.arange(0, total_t, dt)
        
        # sol = itoSRI2(sde.f, sde.g, x0, tspan) # Runge-Kutta algorithm
        sol = itoEuler(sde.f, sde.g, x0, tspan) # Euler-Maruyama algorithm
        u = sol[:, :xdim]
        v = sol[:, xdim:]

        os.makedirs(data_dir, exist_ok=True)

        if plot:
            # heatmap
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(u[::-1], cmap='jet', vmin=-3, vmax=3, aspect='auto', extent=[0, xdim, 0, total_t])
            ax.set_xlabel('x', fontsize=18)
            ax.set_ylabel('t', fontsize=18)
            ax.set_title('u', fontsize=18)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(f'{data_dir}/heatmap_u_delta2_{delta2}_du_{du}_xdim{xdim}.png', dpi=300)
            plt.close()
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(v[::-1], cmap='jet', vmin=-3, vmax=3, aspect='auto', extent=[0, xdim, 0, total_t])
            ax.set_xlabel('x', fontsize=18)
            ax.set_ylabel('t', fontsize=18)
            ax.set_title('v', fontsize=18)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(f'{data_dir}/heatmap_v_delta2_{delta2}_du_{du}_xdim{xdim}.png', dpi=300)
            plt.close()

            # sample trajectories
            plt.figure(figsize=(10,12))
            ax = plt.subplot(211)
            ax.plot(tspan, u[:, 0])
            ax.set_xlabel('t / s', fontsize=18)
            ax.set_ylabel('u_x0', fontsize=18)
            ax = plt.subplot(212)
            ax.plot(tspan, v[:, 0])
            ax.set_xlabel('t / s', fontsize=18)
            ax.set_ylabel('v_x0', fontsize=18)
            plt.subplots_adjust(hspace=0.3)
            plt.savefig(f'{data_dir}/fhn_delta2_{delta2}_du_{du}_xdim{xdim}.png', dpi=300)
            plt.close()

        sol = np.stack((u,v), axis=0).transpose(1,0,2)  # (time_length, 2, xdim)
        return sol
    
    if save and os.path.exists(f'{data_dir}/origin.npz'): return
    
    trace = []
    for trace_id in tqdm(range(1, trace_num+1)):
        sol = solve_1_trace(trace_id)
        trace.append(sol)
    
    if save: 
        np.savez(f'{data_dir}/origin.npz', trace=trace, dt=dt, T=total_t)
        print(f'save origin data form seed 1 to {trace_num} at {data_dir}/')
    
    return np.array(trace)
# generate_original_data(1, total_t=0.9, dt=0.001, save=False, plot=True, xdim=1)


def generate_dataset_static(trace_num, tau=0., dt=0.001, max_tau=0.01, is_print=False, parallel=False, xdim=1, data_dir='Data/PNAS17_xdim1/data'):

    if os.path.exists(f"{data_dir}/tau_{tau}/train_static.npz") and os.path.exists(f"{data_dir}/tau_{tau}/val_static.npz") and os.path.exists(f"{data_dir}/tau_{tau}/test_static.npz"):
        return
    
    # generate simulation data
    if not os.path.exists(f"{data_dir}/static_{max_tau:.2f}.npz"):
        if is_print: print('generating simulation trajectories:')
        # total_t 略大于 max_tau，后面截取不从0开始，以避免所有数据的初始值相同
        data = generate_original_data(trace_num, total_t=max_tau+1*dt, dt=dt, save=False, plot=False, parallel=parallel, xdim=xdim)
        os.makedirs(data_dir, exist_ok=True)
        np.savez(f"{data_dir}/static_{max_tau:.2f}.npz", data=data)
        if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')
    else:
        data = np.load(f"{data_dir}/static_{max_tau:.2f}.npz")['data']

    if is_print: print(f"\r[{tau}/{max_tau}]", end='')

    # save statistic information
    data_dir = f"{data_dir}/tau_{tau}"
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
        sequences = data_item[:, ::step_length]
        sequences = sequences[:, :2]
        
        # save
        np.savez(data_dir+f'/{item}_static.npz', data=sequences)

        # plot
        plt.figure(figsize=(16,10))
        plt.title(f'{item.capitalize()} Data')
        plt.plot(sequences[:,0,0,0], label='u_x0')
        plt.plot(sequences[:,0,1,0], label='v_x0')
        plt.legend()
        plt.savefig(data_dir+f'/{item}_static_input.pdf', dpi=300)

        plt.figure(figsize=(16,10))
        plt.title(f'{item.capitalize()} Data')
        plt.plot(sequences[:,1,0,0], label='u_x0')
        plt.plot(sequences[:,1,1,0], label='v_x0')
        plt.legend()
        plt.savefig(data_dir+f'/{item}_static_target.pdf', dpi=300)

    
def generate_dataset_slidingwindow(trace_num, tau, sample_num=None, is_print=False, sequence_length=None, xdim=1, data_dir='Data/PNAS17_xdim1/data'):

    if (sequence_length is not None) and os.path.exists(f"{data_dir}/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"{data_dir}/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"{data_dir}/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif (sequence_length is None) and os.path.exists(f"{data_dir}/tau_{tau}/train.npz") and os.path.exists(f"{data_dir}/tau_{tau}/val.npz") and os.path.exists(f"{data_dir}/tau_{tau}/test.npz"):
        return
    
    # load original data
    if is_print: print('loading original trace data:')
    origin_dir = data_dir.replace('data', 'origin')
    tmp = np.load(f"{origin_dir}/origin.npz")
    dt = tmp['dt']
    data = tmp['trace']
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

    # save statistic information
    data_dir = f"{data_dir}/tau_{tau}"
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
                plt.plot(sequences[:,0,0,0], label='u_x0')
                plt.plot(sequences[:,0,1,0], label='v_x0')
                plt.legend()
                plt.savefig(data_dir+f'/{item}_input.pdf', dpi=300)

                plt.figure(figsize=(16,10))
                plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
                plt.plot(sequences[:,sequence_length-1,0,0], label='u_x0')
                plt.plot(sequences[:,sequence_length-1,1,0], label='v_x0')
                plt.legend()
                plt.savefig(data_dir+f'/{item}_target.pdf', dpi=300)
