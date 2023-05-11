# -*- coding: utf-8 -*-
import os
import time
import signal
import argparse
import torch
import numpy as np
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
from multiprocessing import Process
import warnings;warnings.simplefilter('ignore')

from methods import *
from util import set_cpu_num, seed_everything


def Data_Generate(args):
    
    # generate original data
    print('Generating original simulation data')
    if 'PNAS17' not in args.system:
        generate_original_data(args.trace_num, args.total_t, args.dt, save=True, plot=True, parallel=args.parallel)
    else:
        origin_dir = args.data_dir.replace('data', 'origin')
        generate_original_data(args.trace_num, args.total_t, args.dt, save=True, plot=True, parallel=args.parallel, xdim=args.xdim, delta=args.delta, du=args.du, data_dir=origin_dir, init_type=args.init_type, clone=args.clone, noise=args.noise)

    # generate dataset for ID estimating
    print('Generating training data for ID estimating')
    T = np.arange(args.tau_1, args.tau_N+args.dt, args.tau_unit)
    for tau in T:
        tau = round(tau, 3)
        if args.sample_type == 'sliding_window':
            if 'PNAS17' in args.system:
                generate_dataset_slidingwindow_slice(args.trace_num, tau, None, is_print=True, xdim=args.xdim, data_dir=args.data_dir)
        else:
            raise NameError(f"{args.sample_type} not implemented!")
    

def ID_subworker_slice(args, tau, random_seed=729, is_print=False, slice_id=1):
    
    time.sleep(0.1)
    
    seed_everything(random_seed)
    set_cpu_num(args.cpu_num)
    
    sliding_window = True if args.sample_type=='sliding_window' else False
    
    # train
    train_time_lagged_slice(args.system, args.embedding_dim, args.channel_num, args.obs_dim, tau, args.id_epoch, is_print, random_seed, args.data_dir, args.id_log_dir, args.device, args.data_dim, args.lr, args.batch_size, args.enc_net, args.e1_layer_n, sliding_window, slice_id)

    # test and calculating ID
    # test_and_save_embeddings_of_time_lagged_slice(args.system, args.embedding_dim, args.channel_num, args.obs_dim, tau, args.id_epoch, None, is_print, random_seed, args.data_dir, args.id_log_dir, args.device, args.data_dim, args.batch_size, args.enc_net, args.e1_layer_n, sliding_window, args.tau_unit, args.total_t, slice_id)
    test_and_save_embeddings_of_time_lagged_slice(args.system, args.embedding_dim, args.channel_num, args.obs_dim, tau, args.id_epoch, args.id_log_dir + f"slice_id{slice_id}/tau_{tau}/seed{random_seed}", is_print, random_seed, args.data_dir, args.id_log_dir, args.device, args.data_dim, args.batch_size, args.enc_net, args.e1_layer_n, sliding_window, args.tau_unit, args.total_t, slice_id)


def ID_Estimate(args):
    
    print('Estimating the ID per tau')
    
    # id estimate process
    T = np.arange(args.tau_1, args.tau_N+args.dt, args.tau_unit)
    workers = []
    for tau in T:
        tau = round(tau, 3)
        # for slice_id in range(int(args.total_t/1.0)):
        for slice_id in [0]:
            for random_seed in range(1, args.seed_num+1):
                if args.parallel: # multi-process to speed-up
                    is_print = True if len(workers)==0 else False
                    workers.append(Process(target=ID_subworker_slice, args=(args, tau, random_seed, is_print, slice_id), daemon=True))
                    workers[-1].start()
                else:
                    ID_subworker_slice(args, tau, random_seed, True, slice_id)
    # block
    while args.parallel and any([sub.exitcode==None for sub in workers]):
        pass

    plot_id_per_slice(range(int(args.total_t)), np.arange(args.id_epoch-5, args.id_epoch+1, 1), log_dir=args.id_log_dir)

    exit(0)

    # plot ID curve
    [plot_epoch_test_log(round(tau, 3), max_epoch=args.id_epoch+1, log_dir=args.id_log_dir) for tau in T]
    plot_id_per_tau(T, np.arange(args.id_epoch-5, args.id_epoch+1, 1), log_dir=args.id_log_dir)
    
    if 'cuda' in args.device: torch.cuda.empty_cache()
    print('\nID Estimate Over')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ours', help='Model: [ours, lstm, tcn, neural_ode]')
    parser.add_argument('--init_type', type=str, default='grf', help='initialization type: [grf, random]')
    parser.add_argument('--xdim', type=int, default=1, help='space dimension')
    parser.add_argument('--clone', type=int, default=1, help='clone the low-dimension to high-dimension')
    parser.add_argument('--noise', type=int, default=0, help='clone with noise')
    parser.add_argument('--system', type=str, default='2S2F', help='Dynamical System: [1S1F, 1S2F, ToggleSwitch, SignalingCascade, HalfMoon, 2S2F, FHN, SC]')
    parser.add_argument('--channel_num', type=int, default=4, help='Overall featrue number')
    parser.add_argument('--obs_dim', type=int, default=4, help='Obervable feature dimension')
    parser.add_argument('--data_dim', type=int, default=4, help='Overall feature dimension')
    parser.add_argument('--trace_num', type=int, default=200, help='Number of simulation trajectories')
    parser.add_argument('--total_t', type=float, default=5.1, help='Time length of each simulation trajectories')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step of each simulation trajectories')
    parser.add_argument('--delta', type=float, default=0.0, help='nosie coefficient')
    parser.add_argument('--du', type=float, default=0.5, help='duffusion coefficient')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--tau_unit', type=float, default=0.1, help='Unit time step of Time Scale')
    parser.add_argument('--tau_1', type=float, default=0.1, help='Lower boundary of Time Scale')
    parser.add_argument('--tau_N', type=float, default=3.0, help='Upper boundary of Time Scale')
    parser.add_argument('--tau_s', type=float, default=0.8, help='Approprate time scale for fast-slow separation')  
    parser.add_argument('--embedding_dim', type=int, default=2, help='Embedding dimensionality of mutual information vector')             
    parser.add_argument('--slow_dim', type=int, default=2, help='Intrinsic dimension of slow dynamics')             
    parser.add_argument('--koopman_dim', type=int, default=4, help='Dimension of Koopman invariable space')
    parser.add_argument('--enc_net', type=str, default='MLP', help='Network type of the Encoder epsilon_1')
    parser.add_argument('--e1_layer_n', type=int, default=3, help='Layer num of the Encoder epsilon_1')
    parser.add_argument('--id_epoch', type=int, default=100, help='Max training epoch of ID-driven Time Scale Selection')
    parser.add_argument('--learn_epoch', type=int, default=100, help='Max training epoch of Fast-Slow Learning')
    parser.add_argument('--baseline_epoch', type=int, default=100, help='Max training epoch of Baseline Algorithm')
    parser.add_argument('--seed_num', type=int, default=10, help='Multiple random seed for average')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--cpu_num', type=int, default=-1, help='Limit the available cpu number of each sub-processing, -1 means no limit')
    parser.add_argument('--parallel', help='Parallel running the whole pipeline by multi-processing', action='store_true')
    parser.add_argument('--data_dir', type=str, default='Data/2S2F/data/')
    parser.add_argument('--id_log_dir', type=str, default='logs/2S2F/TimeSelection/')
    parser.add_argument('--learn_log_dir', type=str, default='logs/2S2F/LearnDynamics/')
    parser.add_argument('--baseline_log_dir', type=str, default='logs/2S2F/LSTM/')
    parser.add_argument('--result_dir', type=str, default='Results/2S2F/')
    parser.add_argument('--plot_corr', help='Plot the correlation and auto-correlation of data variables', action='store_true')
    parser.add_argument('--plot_mi', help='Plot the mutual information of data variables', action='store_true')
    parser.add_argument('--sample_type', help='The sample type when generating ID-esitimate dataset, sliding wnidow or static', type=str, default='sliding_window')
    args = parser.parse_args()

    if 'PNAS17' in args.system:
        from Data.generator_pnas17 import generate_dataset_slidingwindow_slice, generate_original_data
        from util.plot_pnas17 import plot_epoch_test_log, plot_id_per_tau, plot_id_per_slice

    if not args.parallel and args.cpu_num==1:
        print('Not recommand to limit the cpu num when non-parallellism!')
    
    if args.enc_net != 'MLP' and args.e1_layer_n == 2:
        print('RNN layer num should be larger than 2!')

    # ctrl c
    def term_sig_handler(signum, frame):
        exit(0)
    signal.signal(signal.SIGINT, term_sig_handler)
    
    # main pipeline
    Data_Generate(args)

    if args.model == 'ours':
        ID_Estimate(args)