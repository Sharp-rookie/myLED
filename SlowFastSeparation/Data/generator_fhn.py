import os
import numpy as np
from tqdm import tqdm
import scienceplots
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
import warnings;warnings.simplefilter('ignore')

from .lattice_boltzmann import run_lb_fhn_ic


def plot_trace_heatmap(rho_act_all, rho_in_all, x, t_vec, total_t, dt):

    for ic in range(len(rho_act_all)):

        print(f'\rplot trace heatmap [{ic+1}/{len(rho_act_all)}]', end='')

        os.makedirs(f"Data/FHN/origin/{ic}/", exist_ok=True)
            
        rho_act = rho_act_all[ic]
        rho_in = rho_in_all[ic]
        for index, item in enumerate(['act', 'in']):
            N_end = int(total_t / dt)
            X = x
            Y = t_vec[:N_end]
            X, Y = np.meshgrid(X, Y)
            Z = rho_act[:N_end] if index==0 else rho_in[:N_end]
            
            # Picture 1
            plt.figure(figsize=(12,10))
            ax = plt.subplot(111, projection='3d')
            # plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rasterized=True)
            ax.set_xlabel(r"$x$", labelpad=20)
            ax.set_ylabel(r"$t$", labelpad=20)
            ax.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax.set_zlabel(r"$u(x,t)$" if index==0 else r"$v(x,t)$", rotation=0, labelpad=20)
            # ax.set_zlim(-1.5, 1.5)
            # fig.colorbar(surf, orientation="horizontal")
            ax.invert_xaxis()
            ax.view_init(elev=30., azim=30.) # view direction: elve=vertical angle ,azim=horizontal angle
            plt.savefig("Data/FHN/origin/{:}/surface_{:}.pdf".format(ic, item), dpi=300)
            plt.close()

            # Picture 2
            fig = plt.figure()
            ax = fig.gca()
            mp = ax.contourf(X, Y, Z, 100, cmap=plt.get_cmap("seismic"),zorder=-9)
            ax.set_ylabel(r"$t$")
            ax.set_xlabel(r"$x$")
            fig.colorbar(mp)
            plt.gca().set_rasterization_zorder(-1)
            plt.savefig("Data/FHN/origin/{:}/contourf_{:}.pdf".format(ic, item), bbox_inches="tight", dpi=300)
            plt.close()


def plot_trace_gif(rho_act_all, rho_in_all):

    for ic in range(len(rho_act_all)):

        print(f'\rplot trace gif [{ic+1}/{len(rho_act_all)}]', end='')

        os.makedirs(f"Data/FHN/origin/{ic}/", exist_ok=True)

        rho_act = rho_act_all[ic][::500]
        rho_in = rho_in_all[ic][::500]
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        line, = ax.plot(np.linspace(0,20,101), rho_act[0], color='green')
        def update(i):
            label = 'timestep {0}'.format(i)
            line.set_ydata(rho_act[i])
            ax.set_title(f't = {i*5}s')
            ax.set_ylabel('activator')
            ax.set_xlabel('x')
            return line, ax
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, 200), interval=1)
        anim.save(f'Data/FHN/origin/{ic}/activator.gif', dpi=100, writer='pillow')

        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        line, = ax.plot(np.linspace(0,20,101), rho_in[0], color='red')
        def update(i):
            label = 'timestep {0}'.format(i)
            line.set_ydata(rho_in[i])
            ax.set_title(f't = {i*5}s')
            ax.set_ylabel('inhibitor')
            ax.set_xlabel('x')
            return line, ax
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, 200), interval=1)
        anim.save(f'Data/FHN/origin/{ic}/inhibitor.gif', dpi=100, writer='pillow')


def generate_original_data(trace_num, total_t=1000.1, dt=0.01, save=True, plot=False, parallel=False):
    
    if save and os.path.exists('Data/FHN/origin/origin.npz'): return
    
    trace = []

    file_names = ["y00", "y01", "y02", "y03", "y04", "y05"]

    rho_act_all = []
    rho_in_all = []
    for f_id, file_name in enumerate(file_names):

        # load inital-condition file
        rho_act_0 = np.loadtxt(f"Data/FHN_ICs/{file_name}u.txt", delimiter="\n")
        rho_in_0 = np.loadtxt(f"Data/FHN_ICs/{file_name}v.txt", delimiter="\n")
        x = np.loadtxt("Data/FHN_ICs/y0x.txt", delimiter="\n")

        # hyper-params
        epsilon = 0.01
        a1 = 3.

        # simulate by LBM
        rho_act, rho_in, t_vec, _, _, _, _, dt, _, _, _, x, _, _, a0, a1, _, _, total_t = run_lb_fhn_ic(f_id+1, len(file_names), rho_act_0, rho_in_0, total_t, dt, epsilon, a1)
        
        # record
        rho_act_all.append(rho_act)
        rho_in_all.append(rho_in)
    
    trace = np.concatenate([rho_act_all, rho_in_all], axis=-1).reshape(len(file_names),-1,2,101)

    if plot:
        print()
        plot_trace_heatmap(trace[:,:,0,:], trace[:,:,1,:], x, t_vec, total_t, dt)
        print()
        plot_trace_gif(trace[:,:,0,:], trace[:,:,1,:])
    
    if save: 
        os.makedirs('Data/FHN/origin', exist_ok=True)
        np.savez('Data/FHN/origin/origin.npz', trace=trace, dt=dt, T=total_t)

    return trace
# generate_original_data(1, total_t=1000.1, dt=0.01, save=False, plot=True)


# def generate_dataset_static(trace_num, tau=0., dt=0.01, max_tau=5., is_print=False, parallel=False):

#     if os.path.exists(f"Data/FHN/data/tau_{tau}/train_static.npz") and os.path.exists(f"Data/FHN/data/tau_{tau}/val_static.npz") and os.path.exists(f"Data/FHN/data/tau_{tau}/test_static.npz"):
#         return
    
#     # generate simulation data
#     if not os.path.exists("Data/FHN/data/static.npz"):
#         if is_print: print('generating simulation trajectories:')
#         data = generate_original_data(trace_num, total_t=max_tau+dt, dt=dt, save=False, plot=False, parallel=parallel)
#         data = data[:,:,np.newaxis] # add channel dim
#         np.savez("Data/FHN/data/static.npz", data=data)
#         if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')
#     else:
#         data = np.load("Data/FHN/data/static.npz")['data']

#     if is_print: print(f"\r[{tau}/{max_tau}]", end='')

#     # save statistic information
#     data_dir = f"Data/FHN/data/tau_{tau}"
#     os.makedirs(data_dir, exist_ok=True)
#     np.savetxt(data_dir + "/data_mean_static.txt", np.mean(data, axis=(0,1)))
#     np.savetxt(data_dir + "/data_std_static.txt", np.std(data, axis=(0,1)))
#     np.savetxt(data_dir + "/data_max_static.txt", np.max(data, axis=(0,1)))
#     np.savetxt(data_dir + "/data_min_static.txt", np.min(data, axis=(0,1)))
#     np.savetxt(data_dir + "/tau_static.txt", [tau]) # Save the timestep

#     ##################################
#     # Create [train,val,test] dataset
#     ##################################
#     assert trace_num<=6, f"FHN trace_num should not larger than 6, but trace_num={trace_num}"
#     train_num = 3
#     val_num = 1
#     test_num = 2
#     trace_list = {'train':range(train_num), 'val':range(train_num,train_num+val_num), 'test':range(train_num+val_num,train_num+val_num+test_num)}
#     for item in ['train','val','test']:
                
#         # select trace num
#         data_item = data[trace_list[item]]

#         # subsampling
#         step_length = int(tau/dt) if tau!=0. else 1

#         assert step_length<data_item.shape[1], f"Tau {tau} is larger than the simulation time length{dt*data_item.shape[1]}"
#         sequences = data_item[:, ::step_length]
#         sequences = sequences[:, :2]
        
#         # save
#         np.savez(data_dir+f'/{item}_static.npz', data=sequences)

#         # plot
#         plt.figure(figsize=(16,10))
#         plt.title(f'{item.capitalize()} Data')
#         plt.plot(sequences[:,0,0,0], label='u')
#         plt.plot(sequences[:,0,0,1], label='v')
#         plt.legend()
#         plt.savefig(data_dir+f'/{item}_static_input.pdf', dpi=300)

#         plt.figure(figsize=(16,10))
#         plt.title(f'{item.capitalize()} Data')
#         plt.plot(sequences[:,1,0,0], label='u')
#         plt.plot(sequences[:,1,0,1], label='v')
#         plt.legend()
#         plt.savefig(data_dir+f'/{item}_static_target.pdf', dpi=300)

    
def generate_dataset_slidingwindow(trace_num, tau, sample_num=None, is_print=False, sequence_length=None, x_num=101):

    if (sequence_length is not None) and os.path.exists(f"Data/FHN_{x_num}/data/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/FHN_{x_num}/data/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/FHN_{x_num}/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif (sequence_length is None) and os.path.exists(f"Data/FHN_{x_num}/data/tau_{tau}/train.npz") and os.path.exists(f"Data/FHN_{x_num}/data/tau_{tau}/val.npz") and os.path.exists(f"Data/FHN_{x_num}/data/tau_{tau}/test.npz"):
        return
    
    # load original data
    if is_print: print('loading original trace data:')
    tmp = np.load(f"Data/FHN/origin/origin.npz")
    dt = tmp['dt']
    step = int(101/(x_num-1+1e-5))
    data = np.array(tmp['trace'])[:trace_num,:,:,np.arange(0,x_num*step, step)] # (trace_num, time_length, channel, feature_num)
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

    # save statistic information
    data_dir = f"Data/FHN_{x_num}/data/tau_{tau}"
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
    assert trace_num<=6, f"FHN trace_num should not larger than 6, but trace_num={trace_num}"
    train_num = 3
    val_num = 1
    test_num = 2
    trace_list = {'train':range(train_num), 'val':range(train_num,train_num+val_num), 'test':range(train_num+val_num,train_num+val_num+test_num)}
    for item in ['train','val','test']:
                
        # select trace num
        N_TRACE = len(trace_list[item])
        data_item = data[trace_list[item]]

        # space downsample
        print(np.shape(data_item))

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
                plt.figure(figsize=(24,10))
                ax1 = plt.subplot(1,2,1)
                ax2 = plt.subplot(1,2,2)
                for i in range(x_num):
                    ax1.plot(sequences[:,0,0,i], label=f'u_{i}')
                    ax2.plot(sequences[:,0,1,i], label=f'v_{i}')
                ax1.legend()
                ax2.legend()
                plt.savefig(data_dir+f'/{item}_input.pdf', dpi=300)

                plt.figure(figsize=(24,10))
                ax1 = plt.subplot(1,2,1)
                ax2 = plt.subplot(1,2,2)
                for i in range(x_num):
                    ax1.plot(sequences[:,0,0,i], label=f'u_{i}')
                    ax2.plot(sequences[:,0,1,i], label=f'v_{i}')
                ax1.legend()
                ax2.legend()
                plt.savefig(data_dir+f'/{item}_target.pdf', dpi=300)
