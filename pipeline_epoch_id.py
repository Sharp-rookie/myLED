# -*- coding: utf-8 -*-
import os
import time
import glob
import torch
import shutil
import traceback
import numpy as np
from tqdm import tqdm
from munch import munchify
import matplotlib.pyplot as plt
from multiprocessing import Process, JoinableQueue
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings;warnings.simplefilter('ignore')

from utils.gillespie import generate_origin
from utils.data_process import time_discretization
from utils.pnas_dataset import PNASDataset, scaler
from utils.pnas_model import PNAS_VisDynamicsModel
from utils.intrinsic_dimension import eval_id_latent
from utils.common import set_cpu_num, load_config, rm_mkdir


def generate_original_data(trace_num, total_t):

    seed_everything(trace_num+729)

    os.makedirs('Data/origin', exist_ok=True)

    # -------------------- generate original data by gillespie algorithm --------------------
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/origin/{seed}/origin.npz'):
            subprocess.append(Process(target=generate_origin, args=(total_t, seed,), daemon=True))
            subprocess[-1].start()
            print(f'\rStart process[seed={seed}] for origin data' + ' '*30)
        else:
            print(f'\rOrigin data [seed={seed}] existed' + ' '*30)
    while any([subp.exitcode == None for subp in subprocess]):
        pass
    
    # ----------- time discretization by time-forward NearestNeighbor interpolate -----------
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/origin/{seed}/data.npz'):
            subprocess.append(Process(target=time_discretization, args=(seed, total_t,), daemon=True))
            subprocess[-1].start()
            print(f'\rStart process[seed={seed}] for time-discrete data' + ' '*30)
        else:
            print(f'\rTime discrete data [seed={seed}] existed' + ' '*30)
    while any([subp.exitcode == None for subp in subprocess]):
        pass

    print(f'save origin data form seed 1 to {trace_num} at Data/origin/')


def generate_tau_data(trace_num, tau, sample_num=None):

    if os.path.exists(f"Data/data/tau_{tau}/train.npz") and os.path.exists(f"Data/data/tau_{tau}/val.npz") and os.path.exists(f"Data/data/tau_{tau}/test.npz"):
        return

    # -------------------------------- 1_load_original_data -------------------------------- 

    print('loading original data ...')
    data = []
    for id in range(1, trace_num+1):
        tmp = np.load(f"Data/origin/{id}/data.npz")
        X = np.array(tmp['X'])[:, np.newaxis]
        Y = np.array(tmp['Y'])[:, np.newaxis]
        Z = np.array(tmp['Z'])[:, np.newaxis]

        trace = np.concatenate((X, Y, Z),axis=1)
        data.append(trace)
    data = np.array(data)

    # -------------------------------- 2_create_tau_data -------------------------------- 

    # subsampling
    dt = tmp['dt']
    subsampling = int(tau/dt) if tau!=0. else 1
    data = data[:, ::subsampling]
    print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, feature_num)')

    # Save statistic information
    data_dir = f"Data/data/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    diff = np.diff(data, axis=1)
    np.savetxt(data_dir + "/diff_mean.txt", np.mean(diff, axis=(0,1)))
    np.savetxt(data_dir + "/diff_std.txt", np.std(diff, axis=(0,1)))
    np.savetxt(data_dir + "/diff_max.txt", np.max(diff, axis=(0,1)))
    np.savetxt(data_dir + "/diff_min.txt", np.min(diff, axis=(0,1)))
    np.savetxt(data_dir + "/data_mean.txt", np.mean(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_std.txt", np.std(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_max.txt", np.max(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_min.txt", np.min(data, axis=(0,1)))
    np.savetxt(data_dir + "/tau.txt", [tau]) # Save the timestep

    # single-sample time steps for train
    sequence_length = 2 if tau != 0. else 1

    #######################
    # Create train data
    #######################

    # select 10 traces for train
    N_TRAIN = len([0])
    data_train = data[[0]]

    # select sliding window index from 2 trace
    idxs_timestep = []
    idxs_ic = []
    for ic in range(N_TRAIN):
        seq_data = data_train[ic]
        idxs = np.arange(0, np.shape(seq_data)[0]-sequence_length, 1)
        for idx_ in idxs:
            idxs_ic.append(ic)
            idxs_timestep.append(idx_)

    # generator train dataset
    sequences = []
    for bn in range(len(idxs_timestep)):
        idx_ic = idxs_ic[bn]
        idx_timestep = idxs_timestep[bn]
        tmp = data_train[idx_ic, idx_timestep:idx_timestep+sequence_length]
        sequences.append(tmp)

    sequences = np.array(sequences) 
    print(f'tau[{tau}]', "original train dataset", np.shape(sequences))

    # keep the length of sequences is equal to sample_num
    if sample_num is not None:
        repeat_num = int(np.floor(N_TRAIN*sample_num/len(sequences)))
        idx = np.random.choice(range(len(sequences)), N_TRAIN*sample_num-len(sequences)*repeat_num, replace=False)
        idx = np.sort(idx)
        tmp1 = sequences[idx]
        tmp2 = None
        for i in range(repeat_num):
            if i == 0:
                tmp2 = sequences
            else:
                tmp2 = np.concatenate((tmp2, sequences), axis=0)
        sequences = tmp1 if tmp2 is None else np.concatenate((tmp1, tmp2), axis=0)
    print(f'tau[{tau}]', "processed train dataset", np.shape(sequences))

    # save train dataset
    np.savez(data_dir+'/train.npz', data=sequences)

    # plot
    plt.figure(figsize=(16,10))
    plt.title('Train Data' + f' | sample_num[{sample_num if sample_num is not None else len(sequences)}]')
    ax1 = plt.subplot(3,1,1)
    ax1.set_title('X')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:, 0, 0])
    plt.xlabel('time / s')

    ax2 = plt.subplot(3,1,2)
    ax2.set_title('Y')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:, 0, 1])
    plt.xlabel('time / s')

    ax3 = plt.subplot(3,1,3)
    ax3.set_title('Z')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:, 0, 2])
    plt.xlabel('time / s')

    plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.95, 
                    top=0.95, 
                    hspace=0.35, 
                    )
    plt.savefig(data_dir+'/train.jpg', dpi=300)

    #######################
    # Create valid data
    #######################

    # select 1 traces for val
    N_VAL = len([10])
    data_train = data[[10]]

    # select sliding window index from 1 trace
    idxs_timestep = []
    idxs_ic = []
    for ic in range(N_VAL):
        seq_data = data_train[ic]
        idxs = np.arange(0, np.shape(seq_data)[0]-sequence_length, 1)
        for idx_ in idxs:
            idxs_ic.append(ic)
            idxs_timestep.append(idx_)

    # generator train dataset
    sequences = []
    for bn in range(len(idxs_timestep)):
        idx_ic = idxs_ic[bn]
        idx_timestep = idxs_timestep[bn]
        tmp = data_train[idx_ic, idx_timestep:idx_timestep+sequence_length]
        sequences.append(tmp)

    sequences = np.array(sequences)
    print(f'tau[{tau}]', "orginal val dataset", np.shape(sequences))

    # keep the length of sequences is equal to sample_num
    if sample_num is not None:
        repeat_num = int(np.floor(N_VAL*sample_num/len(sequences)))
        idx = np.random.choice(range(len(sequences)), N_VAL*sample_num-len(sequences)*repeat_num, replace=False)
        idx = np.sort(idx)
        tmp1 = sequences[idx]
        tmp2 = None
        for i in range(repeat_num):
            if i == 0:
                tmp2 = sequences
            else:
                tmp2 = np.concatenate((tmp2, sequences), axis=0)
        sequences = tmp1 if tmp2 is None else np.concatenate((tmp1, tmp2), axis=0)
    print(f'tau[{tau}]', "processed val dataset", np.shape(sequences))
    
    # save train dataset
    np.savez(data_dir+'/val.npz', data=sequences)

    # plot
    plt.figure(figsize=(16,10))
    plt.title('Val Data' + f' | sample_num[{sample_num if sample_num is not None else len(sequences)}]')
    ax1 = plt.subplot(3,1,1)
    ax1.set_title('X')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:, 0, 0])
    plt.xlabel('time / s')

    ax2 = plt.subplot(3,1,2)
    ax2.set_title('Y')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:, 0, 1])
    plt.xlabel('time / s')

    ax3 = plt.subplot(3,1,3)
    ax3.set_title('Z')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:, 0, 2])
    plt.xlabel('time / s')

    plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.95, 
                    top=0.95, 
                    hspace=0.35, 
                    )
    plt.savefig(data_dir+'/val.jpg', dpi=300)

    #######################
    # Create test data
    #######################

    if os.path.exists(data_dir+"/test.npz"):
        return

    # select 1 traces for train
    N_TEST = len([3])
    data_train = data[[3]]

    # select sliding window index from 1 trace
    idxs_timestep = []
    idxs_ic = []
    for ic in range(N_TEST):
        seq_data = data_train[ic]
        idxs = np.arange(0, np.shape(seq_data)[0]-sequence_length, 1)
        for idx_ in idxs:
            idxs_ic.append(ic)
            idxs_timestep.append(idx_)

    # generator train dataset
    sequences = []
    for bn in range(len(idxs_timestep)):
        idx_ic = idxs_ic[bn]
        idx_timestep = idxs_timestep[bn]
        tmp = data_train[idx_ic, idx_timestep:idx_timestep+sequence_length]
        sequences.append(tmp)

    sequences = np.array(sequences) 
    print(f'tau[{tau}]', "original test dataset", np.shape(sequences))

    # keep the length of sequences is equal to sample_num
    if sample_num is not None:
        repeat_num = int(np.floor(N_TEST*sample_num/len(sequences)))
        idx = np.random.choice(range(len(sequences)), N_TEST*sample_num-len(sequences)*repeat_num, replace=False)
        idx = np.sort(idx)
        tmp1 = sequences[idx]
        tmp2 = None
        for i in range(repeat_num):
            if i == 0:
                tmp2 = sequences
            else:
                tmp2 = np.concatenate((tmp2, sequences), axis=0)
        sequences = tmp1 if tmp2 is None else np.concatenate((tmp1, tmp2), axis=0)
    print(f'tau[{tau}]', "processed test dataset", np.shape(sequences))

    # save train dataset
    np.savez(data_dir+'/test.npz', data=sequences)

    # plot
    plt.figure(figsize=(16,10))
    plt.title('Val Data' + f' | sample_num[{sample_num if sample_num is not None else len(sequences)}]')
    ax1 = plt.subplot(3,1,1)
    ax1.set_title('X')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:, 0, 0])
    plt.xlabel('time / s')

    ax2 = plt.subplot(3,1,2)
    ax2.set_title('Y')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:, 0, 1])
    plt.xlabel('time / s')

    ax3 = plt.subplot(3,1,3)
    ax3.set_title('Z')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:, 0, 2])
    plt.xlabel('time / s')

    plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.95, 
                    top=0.95, 
                    hspace=0.35, 
                    )
    plt.savefig(data_dir+'/test.jpg', dpi=300)


def pnas_main(random_seed, tau):
    cfg = load_config(filepath='config.yaml')
    cfg = munchify(cfg)
    cfg.log_dir += str(tau)
    cfg.data_filepath += str(tau)
    cfg.seed = random_seed

    seed_everything(cfg.seed)
    set_cpu_num(cfg.cpu_num)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])
    
    model = PNAS_VisDynamicsModel(
        lr=cfg.lr,
        seed=cfg.seed,
        if_cuda=cfg.if_cuda,
        if_test=False,
        gamma=cfg.gamma,
        log_dir=log_dir,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        model_name=cfg.model_name,
        data_filepath=cfg.data_filepath,
        dataset=cfg.dataset,
        lr_schedule=cfg.lr_schedule,
        in_channels=cfg.in_channels,
        input_1d_width=cfg.input_1d_width,
    )

    # clear old checkpoint files
    if os.path.exists(log_dir+"/lightning_logs/checkpoints/"):
        shutil.rmtree(log_dir+"/lightning_logs/checkpoints/")

    # define callback for selecting checkpoints during training
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir+"/lightning_logs/checkpoints/",
        filename="{epoch}",
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_top_k=-1
    )

    # define trainer
    trainer = Trainer(gpus=cfg.num_gpus,
                      max_epochs=cfg.epochs,
                      deterministic=True,
                      strategy='ddp_find_unused_parameters_false',
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0,
                      callbacks=checkpoint_callback
    )

    trainer.fit(model)

    print("Best model path:", checkpoint_callback.best_model_path)

def pnas_gather_latent_from_trained_high_dim_model(random_seed, tau, checkpoint_filepath=None):
    
    cfg = load_config(filepath='config.yaml')
    cfg = munchify(cfg)
    cfg.log_dir += str(tau)
    cfg.data_filepath += str(tau)
    cfg.seed = random_seed
    device = torch.device('cuda' if cfg.if_cuda else 'cpu')

    seed_everything(cfg.seed)
    set_cpu_num(cfg.cpu_num)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = PNAS_VisDynamicsModel(
        lr=cfg.lr,
        seed=cfg.seed,
        if_cuda=cfg.if_cuda,
        if_test=False,
        gamma=cfg.gamma,
        log_dir=log_dir,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        model_name=cfg.model_name,
        data_filepath=cfg.data_filepath,
        dataset=cfg.dataset,
        lr_schedule=cfg.lr_schedule,
        in_channels=cfg.in_channels,
        input_1d_width=cfg.input_1d_width,
    )

    # prepare train and test dataset
    kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True} if cfg.if_cuda else {}
    input_scaler = scaler(
                        scaler_type='MinMaxZeroOne',
                        data_min=np.loadtxt(cfg.data_filepath+"/data_min.txt"),
                        data_max=np.loadtxt(cfg.data_filepath+"/data_max.txt"),
            )
    target_scaler = scaler(
                        scaler_type='MinMaxZeroOne',
                        data_min=np.loadtxt(cfg.data_filepath+"/data_min.txt"),
                        data_max=np.loadtxt(cfg.data_filepath+"/data_max.txt"),
            )
    test_dataset = PNASDataset(
        file_path=cfg.data_filepath+'/test.npz', 
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=cfg.test_batch,
                                             shuffle=False,
                                             **kwargs)
    train_dataset = PNASDataset(
        file_path=cfg.data_filepath+'/train.npz', 
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        )
    # prepare test loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=cfg.train_batch,
                                             shuffle=False,
                                             **kwargs)

    # run test forward pass to save the latent vector

    fp = open('test_log.txt', 'a')
    for ep in range(cfg.epochs):

        epoch = ep
        if checkpoint_filepath is not None:
            epoch = ep + 1
            ckpt_path = checkpoint_filepath + f"_pnas_pnas-ae_{cfg.seed}/lightning_logs/checkpoints/" + f'epoch={epoch-1}.ckpt'
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['state_dict'])
        model = model.to(device)
        model.eval()
        model.freeze()

        var_log_dir = log_dir + f'/variables_test_epoch{epoch}'
        rm_mkdir(var_log_dir)

        # Test on test data
        all_latents = []
        test_outputs = np.array([])
        test_targets = np.array([])
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            output, latent = model.model(data.to(device))
            # save the latent vectors
            for idx in range(data.shape[0]):
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                all_latents.append(latent_tmp)
            
            test_outputs = output.cpu().numpy() if not len(test_outputs) else np.concatenate((test_outputs, output.cpu().numpy()), axis=0)
            test_targets = target.cpu().numpy() if not len(test_targets) else np.concatenate((test_targets, target.cpu().numpy()), axis=0)

        # mse
        mse_x = np.average((test_outputs[:,0,0] - test_targets[:,0,0])**2)
        mse_y = np.average((test_outputs[:,0,1] - test_targets[:,0,1])**2)
        mse_z = np.average((test_outputs[:,0,2] - test_targets[:,0,2])**2)
        
        # Test on train data
        train_outputs = np.array([])
        train_targets = np.array([])
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            output, _ = model.model(data.to(device))
            train_outputs = output.cpu().numpy() if not len(train_outputs) else np.concatenate((train_outputs, output.cpu().numpy()), axis=0)
            train_targets = target.cpu().numpy() if not len(train_targets) else np.concatenate((train_targets, target.cpu().numpy()), axis=0)
            
            if batch_idx >= len(test_outputs): break
        # plot (100,1,3)
        test_X = []
        test_Y = []
        test_Z = []
        train_X = []
        train_Y = []
        train_Z = []
        plot_tau = tau if tau!=0.0 else 0.005
        for i in range(len(test_outputs)):
            test_X.append([test_outputs[i,0,0], test_targets[i,0,0]])
            test_Y.append([test_outputs[i,0,1], test_targets[i,0,1]])
            test_Z.append([test_outputs[i,0,2], test_targets[i,0,2]])
            train_X.append([train_outputs[i,0,0], train_targets[i,0,0]])
            train_Y.append([train_outputs[i,0,1], train_targets[i,0,1]])
            train_Z.append([train_outputs[i,0,2], train_targets[i,0,2]])
        plt.figure(figsize=(16,9))
        ax1 = plt.subplot(2,3,1)
        ax1.set_title('test_X')
        plt.plot(np.array([i*plot_tau for i in range(len(test_X))]), np.array(test_X)[:,1], label='true')
        plt.plot(np.array([i*plot_tau for i in range(len(test_X))]), np.array(test_X)[:,0], label='predict')
        plt.xlabel('time / s')
        ax2 = plt.subplot(2,3,2)
        ax2.set_title('test_Y')
        plt.plot(np.array([i*plot_tau for i in range(len(test_X))]), np.array(test_Y)[:,1], label='true')
        plt.plot(np.array([i*plot_tau for i in range(len(test_X))]), np.array(test_Y)[:,0], label='predict')
        plt.xlabel('time / s')
        ax3 = plt.subplot(2,3,3)
        ax3.set_title('test_Z')
        plt.plot(np.array([i*plot_tau for i in range(len(test_X))]), np.array(test_Z)[:,1], label='true')
        plt.plot(np.array([i*plot_tau for i in range(len(test_X))]), np.array(test_Z)[:,0], label='predict')
        plt.xlabel('time / s')
        ax4 = plt.subplot(2,3,4)
        ax4.set_title('train_X')
        plt.plot(np.array([i*plot_tau for i in range(len(train_X))]), np.array(train_X)[:,1], label='true')
        plt.plot(np.array([i*plot_tau for i in range(len(train_X))]), np.array(train_X)[:,0], label='predict')
        plt.xlabel('time / s')
        ax5 = plt.subplot(2,3,5)
        ax5.set_title('train_Y')
        plt.plot(np.array([i*plot_tau for i in range(len(train_X))]), np.array(train_Y)[:,1], label='true')
        plt.plot(np.array([i*plot_tau for i in range(len(train_X))]), np.array(train_Y)[:,0], label='predict')
        plt.xlabel('time / s')
        ax6 = plt.subplot(2,3,6)
        ax6.set_title('train_Z')
        plt.plot(np.array([i*plot_tau for i in range(len(train_X))]), np.array(train_Z)[:,1], label='true')
        plt.plot(np.array([i*plot_tau for i in range(len(train_X))]), np.array(train_Z)[:,0], label='predict')
        plt.xlabel('time / s')
        plt.subplots_adjust(left=0.1,
            right=0.9,
            top=0.9,
            bottom=0.15,
            wspace=0.2,
            hspace=0.35,
        )
        plt.savefig(var_log_dir+"/result.jpg", dpi=300)

        # save latent
        np.save(var_log_dir+'/latent.npy', all_latents)

        # calculae ID
        LB_id = cal_id_latent(tau, random_seed, epoch, 'Levina_Bickel')
        # MiND_id = cal_id_latent(tau, random_seed, epoch, 'MiND_ML')
        # MADA_id = cal_id_latent(tau, random_seed, epoch, 'MADA')
        # PCA_id = cal_id_latent(tau, random_seed, epoch, 'PCA')

        # record
        # fp.write(f"{tau},{random_seed},{mse_x},{mse_y},{mse_z},{epoch},{LB_id},{MiND_id},{MADA_id},{PCA_id}\n")
        fp.write(f"{tau},{random_seed},{mse_x},{mse_y},{mse_z},{epoch},{LB_id},0,0,0\n")
        fp.flush()

        if checkpoint_filepath is None:
            break
        
    fp.close()


def cal_id_latent(tau, random_seed, epoch, method='Levina_Bickel'):

    cfg = load_config(filepath='config.yaml')
    cfg = munchify(cfg)

    log_dir = '_'.join([cfg.log_dir+str(tau),
                        cfg.dataset,
                        cfg.model_name,
                        str(random_seed)])
    
    var_log_dir = log_dir + f'/variables_test_epoch{epoch}'
    eval_id_latent(var_log_dir, method=method)
    dims = np.load(os.path.join(var_log_dir, 'intrinsic_dimension.npy'))

    return np.mean(dims)


def pipeline(trace_num, tau, queue: JoinableQueue):

    time.sleep(1)

    random_seed = None
    try:
        generate_tau_data(trace_num=trace_num, tau=tau, sample_num=100)

        random_seeds = range(1, 2)
        for random_seed in random_seeds:
            # untrained net for ID
            pnas_gather_latent_from_trained_high_dim_model(random_seed, tau, None)
            
            # train
            if not os.path.exists(f'logs/logs_tau{tau}_pnas_pnas-ae_{random_seed}/lightning_logs/checkpoints/epoch=0.ckpt'):
                pnas_main(random_seed, tau)
            
            # trained net for ID
            pnas_gather_latent_from_trained_high_dim_model(random_seed, tau, f"logs/logs_tau{tau}")
            
            queue.put_nowait([f'Part--{tau}--{random_seed}'])
    
    except:
        if random_seed is None:
            queue.put_nowait([f'Data Generate Error--{tau}', traceback.format_exc()])
        else:
            queue.put_nowait([f'Error--{tau}--{random_seed}', traceback.format_exc()])


if __name__ == '__main__':

    # generate original data
    trace_num = 11
    generate_original_data(trace_num=trace_num, total_t=100)
    
    os.system('rm -rf logs/ test_log.txt plot/')

    # start pipeline-subprocess of different tau
    tau_list = np.arange(0., 2.51, 0.1)
    # tau_list = np.arange(0., 1.99, 0.025)
    queue = JoinableQueue()
    subprocesses = []
    for tau in tau_list:
        tau = round(tau, 3)
        subprocesses.append(Process(target=pipeline, args=(trace_num, tau, queue, ), daemon=True))
        subprocesses[-1].start()
        print(f'Start process[tau={tau}]')
    
    # join main-process
    finish_num = 0
    log_fp = open(f'log_tau{tau_list[0]}to{tau_list[-1]}.txt', 'w')
    while finish_num < len(tau_list):

        # listen
        if not queue.empty():
            pkt = queue.get_nowait()
            if 'Part' in pkt[0]:
                tau = float(pkt[0].split("--")[1])
                random_seed = float(pkt[0].split("--")[2])
                log_fp.write(f'Processing[tau={tau}] finish seed {int(random_seed)}\n')
                log_fp.flush()
            elif 'Data' in pkt[0]:
                tau = float(pkt[0].split("--")[1])
                log_fp.write(f'Processing[tau={tau}] error in data-generating\n')
                log_fp.write(str(pkt[1]))
                log_fp.flush()
            elif 'Error' in pkt[0]:
                tau = float(pkt[0].split("--")[1])
                random_seed = float(pkt[0].split("--")[2])
                log_fp.write(f'Processing[tau={tau}] error in seed {int(random_seed)}\n')
                log_fp.write(str(pkt[1]))
                log_fp.flush()
        # check
        kill_list = []
        for subprocess in subprocesses:
            if subprocess.exitcode != None:
                finish_num += 1
                log_fp.write(f"Processing done with exitcode[{subprocess.exitcode}]\n")
                log_fp.flush()
                kill_list.append(subprocess)
        [subprocesses.remove(subprocess) for subprocess in kill_list]
    log_fp.close()