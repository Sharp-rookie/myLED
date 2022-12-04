# -*- coding: utf-8 -*-
import os
import time
import glob
import h5py
import torch
import shutil
import pickle
import traceback
import numpy as np
from tqdm import tqdm
from numpy import loadtxt
from munch import munchify
from multiprocessing import Process, JoinableQueue
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings;warnings.simplefilter('ignore')

from fhn_eval import mkdir
from Data.Utils.run_lb_fhn_ic import *
from fhn_dataset import FHNDataset, scaler
from fhn_model import FHN_VisDynamicsModel
from fhn_intrinsic_dimension import eval_id_latent
from fhn_main import set_cpu_num, load_config, seed


def generate_data(tau):

    # total time steps for train
    total_length = 3000
    
    if not os.path.exists(f"Data/Simulation_Data/lattice_boltzmann_fhn_tau{tau}.pickle"):

        # ------------------------------------ 0_data_gen ------------------------------------- 
        file_names = ["y00", "y01", "y02", "y03", "y04"]

        rho_act_all = []
        rho_in_all = []
        t_vec_all = []
        mom_act_all = []
        mom_in_all = []
        energ_act_all = []
        energ_in_all = []

        for f_id, file_name in enumerate(file_names):
            file_name_act = "Data/InitialConditions/" + file_name + "u.txt"
            rho_act_0 = loadtxt(file_name_act, delimiter="\n")

            file_name_in = "Data/InitialConditions/" + file_name + "v.txt"
            rho_in_0 = loadtxt(file_name_in, delimiter="\n")

            file_name_x = "Data/InitialConditions/y0x.txt"
            x = loadtxt(file_name_x, delimiter="\n")
            tf = tau*(total_length+10)

            rho_act, rho_in, t_vec, mom_act, mom_in, energ_act, energ_in, dt, N, L, dx, x, Dx, Dy, a0, a1, n1, omegas, tf, a0 = run_lb_fhn_ic(f_id, len(file_names), rho_act_0, rho_in_0, tf)

            # Subsampling
            subsampling = int(tau/dt)
            rho_act = rho_act[::subsampling]
            rho_in = rho_in[::subsampling]
            t_vec = t_vec[::subsampling]
            mom_act = mom_act[::subsampling]
            mom_in = mom_in[::subsampling]
            energ_act = energ_act[::subsampling]
            energ_in = energ_in[::subsampling]

            rho_act_all.append(rho_act)
            rho_in_all.append(rho_in)
            t_vec_all.append(t_vec)
            mom_act_all.append(mom_act)
            mom_in_all.append(mom_in)
            energ_act_all.append(energ_act)
            energ_in_all.append(energ_in)

        data = {
            "dt_data":tau,
            "subsampling":subsampling,
            "rho_act_all":rho_act_all,
            "rho_in_all":rho_in_all,
            "t_vec_all":t_vec_all,
            "mom_act_all":mom_act_all,
            "mom_in_all":mom_in_all,
            "energ_act_all":energ_act_all,
            "energ_in_all":energ_in_all,
            "dt":dt,
            "N":N,
            "L":L,
            "dx":dx,
            "x":x,
            "Dx":Dx,
            "Dy":Dy,
            "a0":a0,
            "a1":a1,
            "n1":n1,
            "omegas":omegas,
            "tf":tf,
            "a0":a0,
        }

        os.makedirs("Data/Simulation_Data", exist_ok=True)
        with open(f"Data/Simulation_Data/lattice_boltzmann_fhn_tau{tau}.pickle", "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    
    else:
        print('wait for loading data ...')
        with open(f"Data/Simulation_Data/lattice_boltzmann_fhn_tau{tau}.pickle", "rb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            simdata = pickle.load(file)
            rho_act_all = np.array(simdata["rho_act_all"])
            rho_in_all = np.array(simdata["rho_in_all"])
            del simdata

    # -------------------------------- 2_create_training_data -------------------------------- 
    rho_act_all = np.expand_dims(rho_act_all, axis=2)
    rho_in_all = np.expand_dims(rho_in_all, axis=2)
    sequences_raw = np.concatenate((rho_act_all, rho_in_all), axis=2)
    print('activator shape', np.shape(rho_act_all))
    print('inhibitor shape', np.shape(rho_in_all))
    print('concat shape', np.shape(sequences_raw))

    # Save statistic information
    data_dir_scaler = f"Data/Data/tau_{tau}"
    os.makedirs(data_dir_scaler, exist_ok=True)
    data_max = np.max(sequences_raw, axis=(0,1,3))
    data_min = np.min(sequences_raw, axis=(0,1,3))
    np.savetxt(data_dir_scaler + "/data_max.txt", data_max)
    np.savetxt(data_dir_scaler + "/data_min.txt", data_min)
    np.savetxt(data_dir_scaler + "/tau.txt", [tau]) # Save the timestep

    # single-sample time steps for train
    sequence_length = 2

    batch_size = 256

    #######################
    # Create train data
    #######################

    # select 3 initial-condition(ICs) traces for train
    ICS_TRAIN = [0, 1, 2]
    N_ICS_TRAIN=len(ICS_TRAIN)
    sequences_raw_train = sequences_raw[ICS_TRAIN, :total_length]

    # random select n*batch_size sequence_length-lengthed trace index from 0,1,2 ICs time-series data
    idxs_timestep = []
    idxs_ic = []
    for ic in range(N_ICS_TRAIN):
        seq_data = sequences_raw_train[ic]
        idxs = np.arange(0, np.shape(seq_data)[0]- sequence_length, 1)
        # idxs = np.random.permutation(idxs)
        for idx_ in idxs:
            idxs_ic.append(ic)
            idxs_timestep.append(idx_)
    max_batches = int(np.floor(len(idxs_ic)/batch_size))
    num_sequences = max_batches*batch_size
    print("Number of sequences = {:}/{:}".format(num_sequences, len(idxs_ic)))

    # generator train dataset by random index from 3 ICs
    sequences = []
    for bn in range(num_sequences):
        idx_ic = idxs_ic[bn]
        idx_timestep = idxs_timestep[bn]
        sequence = sequences_raw_train[idx_ic, idx_timestep:idx_timestep+sequence_length]
        sequences.append(sequence)

    sequences = np.array(sequences) 
    print("Train Dataset", np.shape(sequences))

    # save train dataset
    data_dir = f"Data/Data/tau_{tau}/train"
    os.makedirs(data_dir, exist_ok=True)
    hf = h5py.File(data_dir + '/data.h5', 'w')
    # Only a single sequence_example per dataset group
    for seq_num_ in range(np.shape(sequences)[0]):
        data_group = sequences[seq_num_]
        data_group = np.array(data_group)
        gg = hf.create_group('batch_{:010d}'.format(seq_num_))
        gg.create_dataset('data', data=data_group)
    hf.close()


    #######################
    # Create valid data
    #######################

    # select 2 initial-condition traces for val
    ICS_VAL = [3, 4]
    N_ICS_VAL = len(ICS_VAL)
    sequences_raw_val = sequences_raw[ICS_VAL, :total_length]

    # random select n*batch_size sequence_length-lengthed trace index from 3,4 ICs time-series data
    idxs_timestep = []
    idxs_ic = []
    for ic in range(N_ICS_VAL):
        seq_data = sequences_raw_val[ic]
        idxs = np.arange(0, np.shape(seq_data)[0]- sequence_length, 1)
        # idxs = np.random.permutation(idxs)

        for idx_ in idxs:
            idxs_ic.append(ic)
            idxs_timestep.append(idx_)
    max_batches = int(np.floor(len(idxs_ic)/batch_size))
    print("Number of sequences = {:}/{:}".format(max_batches*batch_size, len(idxs_ic)))
    num_sequences = max_batches*batch_size

    # generator val dataset by random index from 2 ICs
    sequences = []
    for bn in range(num_sequences):
        idx_ic = idxs_ic[bn]
        idx_timestep = idxs_timestep[bn]
        sequence = sequences_raw_val[idx_ic, idx_timestep:idx_timestep+sequence_length]
        sequences.append(sequence)
    sequences = np.array(sequences) 
    print("Val Dataset", np.shape(sequences))

    # save val dataset
    data_dir = f"Data/Data/tau_{tau}/val"
    os.makedirs(data_dir, exist_ok=True)
    hf = h5py.File(data_dir + '/data.h5', 'w')
    # Only a single sequence_example per dataset group
    for seq_num_ in range(np.shape(sequences)[0]): # (960, 121, 202)
        data_group = sequences[seq_num_]
        data_group = np.array(data_group)
        gg = hf.create_group('batch_{:010d}'.format(seq_num_))
        gg.create_dataset('data', data=data_group)
    hf.close()


def fhn_main(random_seed):
    cfg = load_config(filepath='config.yaml')
    cfg = munchify(cfg)
    cfg.log_dir += str(tau)
    cfg.data_filepath += str(tau)
    cfg.seed = random_seed

    seed(cfg)
    seed_everything(cfg.seed)
    set_cpu_num(cfg.cpu_num)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])
    
    model = FHN_VisDynamicsModel(
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
        lr_schedule=cfg.lr_schedule
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


def fhn_gather_latent_from_trained_high_dim_model(random_seed, tau, checkpoint_filepath):
    
    cfg = load_config(filepath='config.yaml')
    cfg = munchify(cfg)
    cfg.log_dir += str(tau)
    cfg.data_filepath += str(tau)
    cfg.seed = random_seed
    device = torch.device('cuda' if cfg.if_cuda else 'cpu')

    seed(cfg)
    seed_everything(cfg.seed)
    set_cpu_num(cfg.cpu_num)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = FHN_VisDynamicsModel(
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
        lr_schedule=cfg.lr_schedule
    )

    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath+f"_fhn_fhn-ae_{cfg.seed}/lightning_logs/checkpoints", '*.ckpt'))[0]
    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    model.freeze()

    # prepare train and val dataset
    kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True} if cfg.if_cuda else {}
    data_info_dict = {
                'truncate_data_batches': 4096, 
                'scaler': scaler(
                        scaler_type='MinMaxZeroOne',
                        data_min=np.loadtxt(cfg.data_filepath+"/data_min.txt"),
                        data_max=np.loadtxt(cfg.data_filepath+"/data_max.txt"),
                        channels=1,
                        common_scaling_per_input_dim=0,
                        common_scaling_per_channels=1,  # Common scaling for all channels
                    ), 
                }
    data_info_dict['truncate_data_batches'] = 4096
    val_dataset = FHNDataset(cfg.data_filepath+'/val',
                              data_cache_size=3,
                              data_info_dict=data_info_dict)
    # prepare val loader
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=cfg.val_batch,
                                             shuffle=False,
                                             **kwargs)

    # run val forward pass to save the latent vector for validating the refine network later
    all_latents = []
    var_log_dir = log_dir + '/variables_val'
    mkdir(var_log_dir)
    plot_act_true = []
    plot_act_pred = []
    for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
        output, latent = model.model(data.to(device))
        # save the latent vectors
        for idx in range(data.shape[0]):
            latent_tmp = latent[idx].view(1, -1)[0]
            latent_tmp = latent_tmp.cpu().detach().numpy()
            all_latents.append(latent_tmp)
        
        for i in range(target.shape[0]):
            plot_act_true.append(target[i, 0].unsqueeze(0))
            plot_act_pred.append(output[i, 0].unsqueeze(0))
    
    import matplotlib.pyplot as plt
    os.system('export MPLBACKEND=Agg')
    plot_act_true = torch.cat(plot_act_true, dim=0)[::1]
    plot_act_pred = torch.cat(plot_act_pred, dim=0)[::1]
    dimension = 55
    length = plot_act_true.shape[0]
    plt.figure()
    plt.plot(range(length), plot_act_true[:length, dimension].cpu(), label='True')
    plt.plot(range(length), plot_act_pred[:length, dimension].cpu(), label='Predict')
    plt.legend()
    plt.savefig(log_dir+f"/act_tau{tau}_dimension{dimension}_seed{cfg.seed}.jpg", dpi=300)

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

        eval_id_latent(var_log_dir, if_refine=False, if_all_methods=False)

        dims = np.load(os.path.join(var_log_dir, 'intrinsic_dimension.npy'))
    dims_all.append(dims)
    dim_mean = np.mean(dims_all)
    dim_std = np.std(dims_all)
    
    with open("logs/ID.txt", 'a+b') as fp:
        print(f'tau[{tau}] Mean(std): ' + f'{dim_mean:.4f} (+-{dim_std:.4f})\n')
        fp.write(f'{tau}--{dim_mean:.4f}\n'.encode('utf-8'))
        fp.flush()

def pipeline(tau, queue: JoinableQueue):

    random_seed = None
    try:
        generate_data(tau)

        random_seeds = range(1,10)
        for random_seed in random_seeds:
            if not os.path.exists(f'logs/logs_tau{tau}_fhn_fhn-ae_{random_seed}/act_tau{tau}_dimension55_seed{random_seed}.jpg'):
                fhn_main(random_seed)
                fhn_gather_latent_from_trained_high_dim_model(random_seed, tau, f"logs/logs_tau{tau}")
            queue.put_nowait([f'Part--{tau}--{random_seed}'])
    
        cal_id_latent(tau, random_seeds)
    except:
        if random_seed is None:
            queue.put_nowait([f'Data Generate Error--{tau}', traceback.format_exc()])
        else:
            queue.put_nowait([f'Error--{tau}--{random_seed}', traceback.format_exc()])


if __name__ == '__main__':

    tau_list = np.arange(0.005, 1.0, 0.05)

    # start
    queue = JoinableQueue()
    subprocesses = []
    for tau in tau_list:
        tau = round(tau, 3)
        subprocesses.append(Process(target=pipeline, args=(tau, queue, ), daemon=True))
        subprocesses[-1].start()
        print(f'Start process[tau={tau}]')
        time.sleep(0.1)
    
    # join
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