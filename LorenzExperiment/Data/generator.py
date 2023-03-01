import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pytorch_lightning import seed_everything
import warnings;warnings.simplefilter('ignore')


class CoupledLorenz():
    def __init__(self, sigma, epsilon, a, b, c, cz, r, k):
        self.sigma = sigma
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.c = c
        self.cz = cz
        self.r = r
        self.k = k

    def __call__(self, y0, t):
        
        x, y, z, X, Y, Z = y0

        dx = self.sigma*(y-x) - self.c*(self.a*X+self.k)
        dy = self.r*x - y - x*z + self.c*(self.a*Y+self.k)
        dz = x*y - self.b*z + self.cz*Z

        dX = self.epsilon*self.sigma*(Y-X) - self.c*(x+self.k)
        dY = self.epsilon*(self.r*X-Y-self.a*X*Z) + self.c*(y+self.k)
        dZ = self.epsilon*(self.a*X*Y - self.b*Z) - self.cz*z

        return [dx, dy, dz, dX, dY, dZ]


def generate_original_data(trace_num, total_t=5, dt=0.01, save=True):
    
    def solve_1_trace(trace_id=0, total_t=5, dt=0.01):
        
        seed_everything(trace_id)
        
        y0 = np.random.uniform(-1., 1., 6)

        t  =np.arange(0, total_t, dt)
        c = 0.15
        sol = odeint(CoupledLorenz(10.,0.1,1.,8/3,c,c,28.,0.), y0, t)

        import scienceplots
        plt.style.use(['science'])
        plt.rcParams.update({'font.size':16})
        fig = plt.figure(figsize=(10,21))
        for i in range(2):
            ax = plt.subplot(1, 2, i+1, projection='3d')
            ax.plot(sol[:,0+3*i], sol[:,1+3*i], sol[:,2+3*i], linewidth=1, color='blue' if i==0 else 'red')
            ax.set_xlabel('x' if i==0 else 'X')
            ax.set_ylabel('y' if i==0 else 'Y')
            ax.set_zlabel('z' if i==0 else 'Z')
        plt.subplots_adjust(wspace=0.25)
        os.makedirs('Data/origin/', exist_ok=True)
        plt.savefig(f'Data/origin/lorenz_c{c}.pdf', dpi=300)
        
        return sol
    
    if save and os.path.exists('Data/origin/origin.npz'): return
    
    trace = []
    for trace_id in tqdm(range(1, trace_num+1)):
        sol = solve_1_trace(trace_id, total_t, dt)
        trace.append(sol)
    
    if save: 
        os.makedirs('Data/origin', exist_ok=True)
        np.savez('Data/origin/origin.npz', trace=trace, dt=dt, T=total_t)

    print(f'save origin data form seed 1 to {trace_num} at Data/origin/')

    return trace
generate_original_data(1, total_t=200., dt=0.01, save=False)


# def plot_c3_c4_trajectory():
    
#     c1, c2 = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))

#     omega = 3
#     c3 = np.sin(omega*c1)*np.sin(omega*c2)
#     c4 = 1/((1+np.exp(-omega*c1))*(1+np.exp(-omega*c2)))

#     y0 = [1.,1.,2.,1.5]
#     t  =np.arange(0, 5.1, 0.05)
#     sol = odeint(system_4d, y0, t)
#     c1_trace = sol[:, 0]
#     c2_trace = sol[:, 1]
#     c3_trace = sol[:, 2]
#     c4_trace = sol[:, 3]
    
#     # Picture 1
#     import scienceplots
#     plt.style.use(['science'])
#     for i, c in enumerate([c3, c4]):
#         fig = plt.figure(figsize=(6,6))
#         plt.rcParams.update({'font.size':16})
#         ax = fig.gca(projection='3d')
#         # plot the surface.
#         ax.scatter(c1, c2, c, marker='.', color='k', label=rf'Points on slow-manifold surface')
#         ax.plot(c1_trace, c2_trace, c3_trace, linewidth=2, color="r", label=rf'Solution  trajectory')
#         ax.scatter(c1_trace[::6], c2_trace[::6], c3_trace[::6], linewidth=2, color="b", marker='o')
#         ax.set_xlabel(r"$c_2$", labelpad=10, fontsize=18)
#         ax.set_ylabel(r"$c_1$", labelpad=10, fontsize=18)
#         ax.zaxis.set_rotate_label(False)  # disable automatic rotation
#         ax.set_zlabel(r"$c_3$", labelpad=10, fontsize=18)
#         ax.invert_xaxis()
#         ax.invert_yaxis()
#         ax.grid(False)
#         ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#         ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#         ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#         ax.view_init(elev=25. if i==0 else 15., azim=-100 if i==0 else -15.) # view direction: elve=vertical angle ,azim=horizontal angle
#         plt.tick_params(labelsize=16)
#         ax.xaxis.set_major_locator(plt.MultipleLocator(1))
#         ax.yaxis.set_major_locator(plt.MultipleLocator(1))
#         ax.zaxis.set_major_locator(plt.MultipleLocator(1))
#         plt.legend()
#         plt.subplots_adjust(bottom=0., top=1.)
#         plt.savefig(f"Data/origin/c{2+i+1}.pdf", dpi=300)
#         plt.close()
# plot_c3_c4_trajectory()
    
def generate_dataset(trace_num, tau, sample_num=None, is_print=False, sequence_length=None, neural_ode=False):

    if not neural_ode and (sequence_length is not None) and os.path.exists(f"Data/data/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif not neural_ode and (sequence_length is None) and os.path.exists(f"Data/data/tau_{tau}/train.npz") and os.path.exists(f"Data/data/tau_{tau}/val.npz") and os.path.exists(f"Data/data/tau_{tau}/test.npz"):
        return
    elif neural_ode and (sequence_length is None) and os.path.exists(f"Data/data/tau_{tau}/neural_ode_train.npz") and os.path.exists(f"Data/data/tau_{tau}/neural_ode_val.npz") and os.path.exists(f"Data/data/tau_{tau}/neural_ode_test.npz"):
        return
    
    # load original data
    if is_print: print('loading original trace data:')
    tmp = np.load(f"Data/origin/origin.npz", allow_pickle=True)
    dt = tmp['dt']
    data = np.array(tmp['trace'])[:trace_num,:,np.newaxis] # (trace_num, time_length, channel, feature_num)
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

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
        N_TRACE = len(trace_list[item]) if not neural_ode else 1
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
        if neural_ode:
            for i in range(len(parallel_sequences)):
                parallel_sequences[i] = np.array(parallel_sequences[i])
            parallel_sequences = np.array(parallel_sequences)
            np.savez(data_dir+f'/neural_ode_{item}.npz', data=parallel_sequences[:,:,0]) # （trace_num, time_step, 1, 4）
        else:
            if not seq_none:
                np.savez(data_dir+f'/{item}_{sequence_length}.npz', data=sequences)
            else:
                np.savez(data_dir+f'/{item}.npz', data=sequences)

            # plot
            if seq_none:
                plt.figure(figsize=(16,10))
                plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
                plt.plot(sequences[:,0,0,0], label='c1')
                plt.plot(sequences[:,0,0,1], label='c2')
                plt.plot(sequences[:,0,0,2], label='c3')
                plt.plot(sequences[:,0,0,3], label='c4')
                plt.legend()
                plt.savefig(data_dir+f'/{item}_input.pdf', dpi=300)

                plt.figure(figsize=(16,10))
                plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
                plt.plot(sequences[:,sequence_length-1,0,0], label='c1')
                plt.plot(sequences[:,sequence_length-1,0,1], label='c2')
                plt.plot(sequences[:,sequence_length-1,0,2], label='c3')
                plt.plot(sequences[:,sequence_length-1,0,3], label='c4')
                plt.legend()
                plt.savefig(data_dir+f'/{item}_target.pdf', dpi=300)

            
def generate_informer_dataset(trace_num=1000, total_t=5.1, dt=0.01, sample_num=None):

    # load original data
    simdata = generate_original_data(trace_num=trace_num, total_t=total_t, dt=dt, save=False)
    simdata = np.array(simdata)[:trace_num,:,np.newaxis] # (trace_num, time_length, channel, feature_num)

    for tau in [0.1, 1.0, 5.0]:
        # subsampling
        subsampling = int(tau/dt) if tau!=0. else 1
        data = simdata[:, ::subsampling]
        print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')
        
        import pandas as pd
        data = np.concatenate(data, axis=0)[:,0]
        df = pd.DataFrame(data, columns=['c1','c2','c3','c4'])
        
        t = pd.date_range('2016-07-01 00:00:00', periods=len(df), freq='h')
        df['date'] = t
        df = df[['date','c1','c2','c3','c4']]
        
        df.to_csv(f'tau_{tau}.csv', index=False)

# generate_informer_dataset(trace_num=200, total_t=5.1, dt=0.01, sample_num=None)