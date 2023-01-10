# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import warnings;warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams["text.usetex"] = True
plt.rcParams['xtick.major.pad']='10'
plt.rcParams['ytick.major.pad']='10'
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

from lattice_boltzmann import run_lb_fhn_ic
# from util.plot import plot_contourf_fhn


def plot_original_data(rho_act_all, rho_in_all, x, tf, dt, t_vec_all):

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
    for ic in range(np.shape(rho_act_all)[0]):
        ax1.plot(x, rho_act_all[ic,0,:], label=f"ic[{ic}]", linewidth=2)

    ax1.set_ylabel(r"$u(x,t=0)$")
    ax1.set_xlabel(r"$x$")
    ax1.set_xlim([np.min(x), np.max(x)])
    ax1.set_title("Activator")

    for ic in range(np.shape(rho_in_all)[0]):
        ax2.plot(x, rho_in_all[ic,0,:], label=f"ic[{ic}]", linewidth=2)

    ax2.set_ylabel(r"$v(x,t=0)$")
    ax2.set_xlabel(r"$x$")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax2.set_title("Inhibitor")
    ax2.set_xlim([np.min(x), np.max(x)])
    plt.tight_layout()

    os.makedirs("Data/origin/figures/", exist_ok=True)
    plt.savefig("Data/origin/figures/initial_conditions.jpg", bbox_inches="tight", dpi=300)
    plt.close()


    for ic in tqdm(range(len(rho_act_all))):

        N_end = int(tf / dt)

        rho_act = rho_act_all[ic]
        rho_in = rho_in_all[ic]
        t_vec = t_vec_all[ic]

        subsample_time = 10
        
        os.makedirs("Data/origin/figures/{:}/".format(ic), exist_ok=True)
        
        for index, item in enumerate(['act', 'in']):
            X = x
            Y = t_vec[:N_end]
            Y = Y[::subsample_time]

            X, Y = np.meshgrid(X, Y)

            Z = rho_act[:N_end] if index==0 else rho_in[:N_end]
            Z = Z[::subsample_time]
            
            # Picture 1
            fig = plt.figure(figsize=(12,10))
            ax = fig.gca(projection='3d')
            # plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rasterized=True)
            ax.set_xlabel(r"$x$", labelpad=20)
            ax.set_ylabel(r"$t$", labelpad=20)
            ax.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax.set_zlabel(r"$u(x,t)$", rotation=0, labelpad=20)
            # add a color bar which maps values to colors.
            fig.colorbar(surf, orientation="horizontal")
            ax.invert_xaxis()
            ax.view_init(elev=30., azim=30.) # view direction: elve=vertical angle ,azim=horizontal angle
            plt.savefig("Data/origin/figures/{:}/surface_{:}.jpg".format(ic, item), dpi=300)
            plt.close()

            # Picture 2
            fig = plt.figure()
            ax = fig.gca()
            mp = ax.contourf(X, Y, Z, 100, cmap=plt.get_cmap("seismic"),zorder=-9)
            ax.set_ylabel(r"$t$")
            ax.set_xlabel(r"$x$")
            fig.colorbar(mp)
            plt.gca().set_rasterization_zorder(-1)
            plt.savefig("Data/origin/figures/{:}/contourf_{:}.jpg".format(ic, item), bbox_inches="tight", dpi=300)
            plt.close()


def generate_origin_data(tf=451, dt=0.001):
    
    # u is the Activator; v is the Inhibitor
    
    if os.path.exists("Data/origin/lattice_boltzmann.npz"): return

    file_names = ["y00", "y01", "y02", "y03", "y04", "y05"]

    rho_act_all = []
    rho_in_all = []
    t_vec_all = []
    mom_act_all = []
    mom_in_all = []
    energ_act_all = []
    energ_in_all = []

    for f_id, file_name in enumerate(file_names):
        
        # load inital-condition file
        rho_act_0 = np.loadtxt(f"Data/ICs/{file_name}u.txt", delimiter="\n")
        rho_in_0 = np.loadtxt(f"Data/ICs/{file_name}v.txt", delimiter="\n")
        x = np.loadtxt("Data/ICs/y0x.txt", delimiter="\n")
        
        # simulate by LBM
        rho_act, rho_in, t_vec, mom_act, mom_in, energ_act, energ_in, dt, N, L, dx, x, Dx, Dy, a0, a1, n1, omegas, tf, a0 = run_lb_fhn_ic(f_id, len(file_names), rho_act_0, rho_in_0, tf, dt)

        # record
        rho_act_all.append(rho_act)
        rho_in_all.append(rho_in)
        t_vec_all.append(t_vec)
        mom_act_all.append(mom_act)
        mom_in_all.append(mom_in)
        energ_act_all.append(energ_act)
        energ_in_all.append(energ_in)

    os.makedirs("Data/origin", exist_ok=True)
    np.savez("Data/origin/lattice_boltzmann.npz", rho_act_all=rho_act_all, rho_in_all=rho_in_all, t_vec_all=t_vec_all, dt=dt, x=x, tf=tf)
    
    plot_original_data(np.array(rho_act_all), np.array(rho_in_all), np.array(x), tf, dt, np.array(t_vec_all))
    
    
def analysis_noise_data(tf=451, dt=0.001):

    file_name = "y00"

    rho_act_all = []
    rho_in_all = []
        
    # load inital-condition file
    rho_act_0 = np.loadtxt(f"Data/ICs/{file_name}u.txt", delimiter="\n")
    rho_in_0 = np.loadtxt(f"Data/ICs/{file_name}v.txt", delimiter="\n")
    x = np.loadtxt("Data/ICs/y0x.txt", delimiter="\n")
        
    for f_id in range(2):
        
        if f_id != 0:
            rho_act_0 += np.random.normal(0, 0.1, len(x))
            rho_in_0 += np.random.normal(0, 0.1, len(x))
        
        # simulate by LBM
        rho_act, rho_in, t_vec, mom_act, mom_in, energ_act, energ_in, dt, N, L, dx, x, Dx, Dy, a0, a1, n1, omegas, tf, a0 = run_lb_fhn_ic(f_id, 2, rho_act_0, rho_in_0, tf, dt)

        # record
        rho_act_all.append(rho_act)
        rho_in_all.append(rho_in)
    
    # diff
    rho_act_all.append(rho_act_all[1]-rho_act_all[0])
    rho_in_all.append(rho_in_all[1]-rho_in_all[0])
    
    rho_act_all = np.array(rho_act_all)
    rho_in_all = np.array(rho_in_all)
    
    # plot
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,6))
    for ic in range(len(rho_act_all)-1):
        ax1.plot(x, rho_act_all[ic,0,:], label=f"ic[{ic}]", linewidth=2)
    ax1.set_ylabel(r"$u(x,t=0)$")
    ax1.set_xlabel(r"$x$")
    ax1.set_xlim([np.min(x), np.max(x)])
    ax1.set_title("Activator")

    for ic in range(len(rho_in_all)-1):
        ax2.plot(x, rho_in_all[ic,0,:], label=f"ic[{ic}]", linewidth=2)
    ax2.set_ylabel(r"$v(x,t=0)$")
    ax2.set_xlabel(r"$x$")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax2.set_title("Inhibitor")
    ax2.set_xlim([np.min(x), np.max(x)])
    plt.tight_layout()
    os.makedirs("Data/analyss/", exist_ok=True)
    plt.savefig("Data/analyss/initial_conditions.jpg", bbox_inches="tight", dpi=300)
    plt.close()

    for ic in tqdm(range(len(rho_act_all))):

        rho_act = rho_act_all[ic]
        rho_in = rho_in_all[ic]
                
        os.makedirs(f"Data/analyss/{ic}/", exist_ok=True)
        
        for index, item in enumerate(['act', 'in']):
            N_end = int(tf / dt)
            X = x
            Y = t_vec[:N_end]
            X, Y = np.meshgrid(X, Y)
            Z = rho_act[:N_end] if index==0 else rho_in[:N_end]
            
            # Picture 1
            fig = plt.figure(figsize=(12,10))
            ax = fig.gca(projection='3d')
            # plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rasterized=True)
            ax.set_xlabel(r"$x$", labelpad=20)
            ax.set_ylabel(r"$t$", labelpad=20)
            ax.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax.set_zlabel(r"$u(x,t)$" if index==0 else r"$v(x,t)$", rotation=0, labelpad=20)
            ax.set_zlim(-1.5, 1.5)
            # fig.colorbar(surf, orientation="horizontal")
            ax.invert_xaxis()
            ax.view_init(elev=30., azim=30.) # view direction: elve=vertical angle ,azim=horizontal angle
            plt.savefig("Data/analyss/{:}/surface_{:}.jpg".format(ic, item), dpi=300)
            plt.close()

            # Picture 2
            fig = plt.figure()
            ax = fig.gca()
            mp = ax.contourf(X, Y, Z, 100, cmap=plt.get_cmap("seismic"),zorder=-9)
            ax.set_ylabel(r"$t$")
            ax.set_xlabel(r"$x$")
            fig.colorbar(mp)
            plt.gca().set_rasterization_zorder(-1)
            plt.savefig("Data/analyss/{:}/contourf_{:}.jpg".format(ic, item), bbox_inches="tight", dpi=300)
            plt.close()


def generate_tau_dataset(tau, sample_num=None, is_print=False, sequence_length=None):
    
    if sequence_length is not None and os.path.exists(f"Data/data/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif sequence_length is None and os.path.exists(f"Data/data/tau_{tau}/train.npz") and os.path.exists(f"Data/data/tau_{tau}/val.npz") and os.path.exists(f"Data/data/tau_{tau}/test.npz"):
        return

    # Load original data
    if is_print: print('loading original trace data ...')
    simdata = np.load("Data/origin/lattice_boltzmann.npz", allow_pickle=True)
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
    for mode in ['train','val','test']:
        
        # if os.path.exists(data_dir+f'/{mode}.npz'): continue
        if sequence_length is not None and os.path.exists(f"Data/data/tau_{tau}/{mode}_{sequence_length}.npz"): continue
        elif sequence_length is None and os.path.exists(f"Data/data/tau_{tau}/{mode}.npz"): continue
        
        # select trace num
        N_TRACE = len(trace_list[mode])
        data_mode = data[trace_list[mode]]
        
        # select sliding window index from N trace
        idxs_timestep = []
        idxs_ic = []
        for ic in range(N_TRACE):
            seq_data = data_mode[ic]
            idxs = np.arange(0, np.shape(seq_data)[0]-sequence_length, 1)
            for idx_ in idxs:
                idxs_ic.append(ic)
                idxs_timestep.append(idx_)
            if is_print: print(f'\rtau[{tau}] {mode} data process_1[{ic+1}/{N_TRACE}]', end='')
        
        # generator mode dataset
        sequences = []
        for bn in range(len(idxs_timestep)):
            idx_ic = idxs_ic[bn]
            idx_timestep = idxs_timestep[bn]
            tmp = data_mode[idx_ic, idx_timestep:idx_timestep+sequence_length]
            sequences.append(tmp)
            if is_print: print(f'\rtau[{tau}] {mode} data process_2[{bn+1}/{len(idxs_timestep)}]', end='')
            
        sequences = np.array(sequences) 
        if is_print: print(f'\ntau[{tau}]', f"{mode} dataset", np.shape(sequences), '# (sample_num, time_length, feature_num, feature_space_dim)')
        
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
        if is_print: print(f'tau[{tau}]', f"after process", np.shape(sequences), '# (sample_num, time_length, feature_num, feature_space_dim)')
        
        # save mode dataset
        if sequence_length!=1 and sequence_length!=2:
            np.savez(data_dir+f'/{mode}_{sequence_length}.npz', data=sequences)
        else:
            np.savez(data_dir+f'/{mode}.npz', data=sequences)
        
        # plot
        if sequence_length==1 or sequence_length==2:
            plot_contourf_fhn(data=sequences[:, 0], tau=tau, path=data_dir+f"/{mode}")