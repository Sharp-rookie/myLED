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

    if os.path.exists(f"Data/data/tau_{tau}/train/data.npz") and os.path.exists(f"Data/data/tau_{tau}/val/data.npz"):
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
    print('data shape', data.shape, '# (trace_num, time_length, feature_num)')

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

    if os.path.exists(data_dir+"/train.npz") and os.path.exists(data_dir+"/val.npz"):
        return

    # select 2 traces for train
    N_TRAIN = len([0, 1])
    data_train = data[[0, 1]]

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
    print("Train Dataset", np.shape(sequences))

    # keep the length of sequences is equal to sample_num
    if sample_num is not None:
        repeat_num = int(np.floor(sample_num/len(sequences)))
        idx = np.random.choice(range(len(sequences)), sample_num-len(sequences)*repeat_num, replace=False)
        idx = np.sort(idx)
        tmp1 = sequences[idx]
        tmp2 = None
        for i in range(repeat_num):
            if i == 0:
                tmp2 = np.concatenate((sequences, sequences), axis=0)
            else:
                tmp2 = np.concatenate((tmp2, sequences), axis=0)
        sequences = tmp1 if tmp2 is None else np.concatenate((tmp1, tmp2), axis=0)

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

    if os.path.exists(data_dir+"/val.npz"):
        return

    # select 1 traces for train
    N_TRAIN = len([2])
    data_train = data[[2]]

    # select sliding window index from 1 trace
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
    print("Val Dataset", np.shape(sequences))

    # keep the length of sequences is equal to sample_num
    if sample_num is not None:
        repeat_num = int(np.floor(sample_num/len(sequences)))
        idx = np.random.choice(range(len(sequences)), sample_num-len(sequences)*repeat_num, replace=False)
        idx = np.sort(idx)
        tmp1 = sequences[idx]
        tmp2 = None
        for i in range(repeat_num):
            if i == 0:
                tmp2 = np.concatenate((sequences, sequences), axis=0)
            else:
                tmp2 = np.concatenate((tmp2, sequences), axis=0)
        sequences = tmp1 if tmp2 is None else np.concatenate((tmp1, tmp2), axis=0)

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


def pnas_main(random_seed):
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
        filename="{epoch}_{val_loss}",
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_top_k=1
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


def pnas_gather_latent_from_trained_high_dim_model(random_seed, tau, checkpoint_filepath):
    
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

    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath+f"_pnas_pnas-ae_{cfg.seed}/lightning_logs/checkpoints", '*.ckpt'))[0]
    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    model.freeze()

    # prepare train and val dataset
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
    val_dataset = PNASDataset(
        file_path=cfg.data_filepath+'/val.npz', 
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        )
    # prepare val loader
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=cfg.val_batch,
                                             shuffle=False,
                                             **kwargs)

    # run val forward pass to save the latent vector for validating the refine network later
    all_latents = []
    var_log_dir = log_dir + '/variables_val'
    rm_mkdir(var_log_dir)
    outputs = np.array([])
    targets = np.array([])
    for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
        output, latent = model.model(data.to(device))
        # save the latent vectors
        for idx in range(data.shape[0]):
            latent_tmp = latent[idx].view(1, -1)[0]
            latent_tmp = latent_tmp.cpu().detach().numpy()
            all_latents.append(latent_tmp)
        
        outputs = output.cpu().numpy() if not len(outputs) else np.concatenate((outputs, output.cpu().numpy()), axis=0)
        targets = target.cpu().numpy() if not len(targets) else np.concatenate((targets, target.cpu().numpy()), axis=0)

    # record mse
    with open('val_mse.txt', 'a') as fp:
        mse_x = np.average((outputs[:,0,0] - targets[:,0,0])**2)
        mse_y = np.average((outputs[:,0,1] - targets[:,0,1])**2)
        mse_z = np.average((outputs[:,0,2] - targets[:,0,2])**2)
        fp.write(f"{tau},{random_seed},{mse_x},{mse_y},{mse_z}\n")
        fp.flush()
    
    # plot (999,1,3)
    X = []
    Y = []
    Z = []
    for i in range(len(outputs)):
        X.append([outputs[i,0,0], targets[i,0,0]])
        Y.append([outputs[i,0,1], targets[i,0,1]])
        Z.append([outputs[i,0,2], targets[i,0,2]])
    plt.figure(figsize=(16,4))
    ax1 = plt.subplot(1,3,1)
    ax1.set_title('X')
    plt.plot(np.array([i*tau for i in range(len(X))]), np.array(X)[:,1], label='true')
    plt.plot(np.array([i*tau for i in range(len(X))]), np.array(X)[:,0], label='predict')
    plt.xlabel('time / s')
    ax2 = plt.subplot(1,3,2)
    ax2.set_title('Y')
    plt.plot(np.array([i*tau for i in range(len(X))]), np.array(Y)[:,1], label='true')
    plt.plot(np.array([i*tau for i in range(len(X))]), np.array(Y)[:,0], label='predict')
    plt.xlabel('time / s')
    ax3 = plt.subplot(1,3,3)
    ax3.set_title('Z')
    plt.plot(np.array([i*tau for i in range(len(X))]), np.array(Z)[:,1], label='true')
    plt.plot(np.array([i*tau for i in range(len(X))]), np.array(Z)[:,0], label='predict')
    plt.xlabel('time / s')
    plt.subplots_adjust(left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.15,
        wspace=0.2,
    )
    plt.savefig(log_dir+f"/result.jpg", dpi=300)

    # save latent
    print(f'latent.npy save at: {var_log_dir}/latent.npy')
    np.save(var_log_dir+'/latent.npy', all_latents)


def cal_id_latent(tau, random_seeds):

    cfg = load_config(filepath='config.yaml')
    cfg = munchify(cfg)

    dims_all = []

    for random_seed in random_seeds:
        cfg.seed = random_seed
        log_dir = '_'.join([cfg.log_dir+str(tau),
                            cfg.dataset,
                            cfg.model_name,
                            str(cfg.seed)])
        var_log_dir = log_dir + '/variables_val'

        eval_id_latent(var_log_dir, method='Levina_Bickel')

        dims = np.load(os.path.join(var_log_dir, 'intrinsic_dimension.npy'))
        dims_all.append(dims)
        dim_mean = np.mean(dims_all)
        dim_std = np.std(dims_all)
    
    with open("ID.txt", 'a+b') as fp:
        print(f'tau[{tau}] Mean(std): ' + f'{dim_mean:.4f} (+-{dim_std:.4f})\n')
        fp.write(f'{tau}--{dim_mean:.4f}\n'.encode('utf-8'))
        fp.flush()

def pipeline(trace_num, tau, queue: JoinableQueue):

    time.sleep(1)

    random_seed = None
    try:
        generate_tau_data(trace_num=trace_num, tau=tau, sample_num=100)

        random_seeds = range(1, 21)
        # for random_seed in random_seeds:
        #     if not os.path.exists(f'logs/logs_tau{tau}_pnas_pnas-ae_{random_seed}/result.jpg'):
        #         pnas_main(random_seed)
        #     pnas_gather_latent_from_trained_high_dim_model(random_seed, tau, f"logs/logs_tau{tau}")
        #     queue.put_nowait([f'Part--{tau}--{random_seed}'])
    
        cal_id_latent(tau, random_seeds)
    except:
        if random_seed is None:
            queue.put_nowait([f'Data Generate Error--{tau}', traceback.format_exc()])
        else:
            queue.put_nowait([f'Error--{tau}--{random_seed}', traceback.format_exc()])


if __name__ == '__main__':

    # generate original data
    trace_num = 3
    generate_original_data(trace_num=trace_num, total_t=100)

    # start pipeline-subprocess of different tau
    # tau_list = np.arange(0., 0.01, 0.025)
    tau_list = np.arange(0., 1.99, 0.025)
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