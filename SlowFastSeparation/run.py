# -*- coding: utf-8 -*-
import os
import time
import signal
import argparse
import torch
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
from multiprocessing import Process
import warnings;warnings.simplefilter('ignore')

from methods import *
from util import set_cpu_num, seed_everything, cal_mi_system, plot_mi

    
def ID_subworker(args, tau, data_dir, id_log_dir, random_seed=729, is_print=False):
    
    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(args.cpu_num)
    
    sliding_window = True if args.sample_type=='sliding_window' else False
    
    # train
    train_time_lagged(args.system, args.embedding_dim, args.channel_num, args.obs_dim, tau, args.id_epoch, is_print, random_seed, data_dir, id_log_dir, args.device, args.data_dim, args.lr, args.batch_size, args.enc_net, args.e1_layer_n, sliding_window)

    # test and calculating ID
    # test_and_save_embeddings_of_time_lagged(args.system, args.embedding_dim, args.channel_num, args.obs_dim, tau, args.id_epoch, None, is_print, random_seed, data_dir, id_log_dir, args.device, args.data_dim, args.batch_size, args.enc_net, args.e1_layer_n, sliding_window)
    test_and_save_embeddings_of_time_lagged(args.system, args.embedding_dim, args.channel_num, args.obs_dim, tau, args.id_epoch, id_log_dir+f"tau_{tau}/seed{random_seed}", is_print, random_seed, data_dir, id_log_dir, args.device, args.data_dim, args.batch_size, args.enc_net, args.e1_layer_n, sliding_window)


def learn_subworker(args, n, data_dir, ckpt_path, learn_log_dir, random_seed=729, is_print=False, mode='train'):
    
    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(args.cpu_num)

    sliding_window = True if args.sample_type=='sliding_window' else False

    if mode == 'train':
        # train
        train_slow_extract_and_evolve(args.system, args.embedding_dim, args.channel_num, args.obs_dim, args.tau_s, args.slow_dim, args.koopman_dim, args.tau_unit, n, ckpt_path, is_print, random_seed, args.learn_epoch, data_dir, learn_log_dir, args.device, args.data_dim, args.lr, args.batch_size, args.enc_net, args.e1_layer_n, sliding_window)
    elif mode == 'test':
        # test evolve
        for i in tqdm(range(1, 50+1)):
            delta_t = round(args.tau_unit*i, 3)
            if args.system == '2S2F':
                MSE, RMSE, MAE, MAPE, c1_mae, c2_mae = test_evolve(args.system, args.embedding_dim, args.channel_num, args.obs_dim, args.tau_s, args.learn_epoch, args.slow_dim, args.koopman_dim, delta_t, i, is_print, random_seed, data_dir, learn_log_dir, args.device, args.data_dim, args.batch_size, args.enc_net, args.e1_layer_n)
                with open(args.result_dir+f'ours_evolve_test_{args.tau_s}.txt','a') as f:
                    f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}, {c1_mae}, {c2_mae}\n')
            elif args.system in ['1S1F', '1S2F', 'ToggleSwitch', 'SignalingCascade', 'HalfMoon'] or 'FHN' in args.system:
                MSE, RMSE, MAE, MAPE = test_evolve(args.system, args.embedding_dim, args.channel_num, args.obs_dim, args.tau_s, args.learn_epoch, args.slow_dim, args.koopman_dim, delta_t, i, is_print, random_seed, data_dir, learn_log_dir, args.device, args.data_dim, args.batch_size, args.enc_net, args.e1_layer_n)
                with open(args.result_dir+f'ours_evolve_test_{args.tau_s}.txt','a') as f:
                    f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}\n')
    else:
        raise TypeError(f"Wrong mode of {mode}!")

    
def ID_Estimate(args, data_dir, id_log_dir):
        
    # id estimate process
    T = np.arange(args.tau_1, args.tau_N+args.dt, args.tau_unit)
    workers = []
    for tau in T:
        tau = round(tau, 2)
        for random_seed in range(1, args.seed_num+1):
            if args.parallel: # multi-process to speed-up
                is_print = True if len(workers)==0 else False
                workers.append(Process(target=ID_subworker, args=(args, tau, data_dir, id_log_dir, random_seed, is_print), daemon=True))
                workers[-1].start()
            else:
                ID_subworker(args, tau, data_dir, id_log_dir, random_seed, True)
    # block
    while args.parallel and any([sub.exitcode==None for sub in workers]):
        pass

    # plot ID curve
    # [plot_epoch_test_log(round(tau,2), max_epoch=args.id_epoch+1, log_dir=id_log_dir) for tau in T]
    plot_id_per_tau(T, np.arange(args.id_epoch-200, args.id_epoch+1, 1), log_dir=id_log_dir)
    

def Learn_Slow_Fast(args, data_dir, ckpt_path, learn_log_dir, mode='train'):
    
    os.makedirs(args.result_dir, exist_ok=True)

    assert args.tau_s>args.tau_1, r"$\tau_s$ must larger than $\tau_1$!"
    
    # slow evolve sub-process
    n = int(args.tau_s/args.tau_unit)
    workers = []
    for random_seed in range(1, args.seed_num+1):
        if args.parallel:
            # is_print = True if len(workers)==0 else False
            is_print = False
            workers.append(Process(target=learn_subworker, args=(args, n, data_dir, ckpt_path, learn_log_dir, random_seed, is_print, mode), daemon=True))
            workers[-1].start()
        else:
            raise NotImplementedError("Not support single process!")

    return workers


def local_ID_estimate(args, u_bound, v_bound, u_step, v_step):

    assert args.trace_num>=500, "trace_num should be large in FHN Time Selecting Process!"

    if os.path.exists(args.id_log_dir+f'u0={u:.2f}_v0={v:.2f}/id_per_tau.pdf'): return

    # generate local dataset
    print('Generating local dataset...')
    T = np.arange(args.tau_1, args.tau_N+args.dt, args.tau_unit)
    for tau in T:
        tau = round(tau, 2)
        if args.sample_type=='sliding_window':
            generate_dataset_sliding_window(tau, u_bound, v_bound, args.total_t, args.dt, args.trace_num, u_step, v_step, args.data_dir)
        elif args.sample_type=='static':
            generate_dataset_static(tau, u_bound, v_bound, args.total_t, args.dt, args.trace_num, u_step, v_step, args.data_dir)
        else:
            raise NotImplementedError
    
    # id estimate process
    data_dir = args.data_dir + f'u0={u_bound:.2f}_v0={v_bound:.2f}/'
    id_log_dir = args.id_log_dir + f'u0={u_bound:.2f}_v0={v_bound:.2f}/'
    ID_Estimate(args, data_dir, id_log_dir)


def local_Dynamic_Learn(args, trace_num, u_bound, v_bound, u_step, v_step):

    assert trace_num<=100, "trace_num should be small in FHN Dynamic Learning Process!"

    # generate local dataset
    print(f'\nLearning slow-fast dynamics [{u_bound},{v_bound}]')
    n = int(args.tau_s/args.tau_unit)
    generate_dataset_sliding_window(args.tau_unit, u_bound, v_bound, args.total_t, args.dt, trace_num, u_step, v_step, args.data_dir, n)
    
    # dynamic learning process
    data_dir = args.data_dir + f'u0={u_bound:.2f}_v0={v_bound:.2f}/'
    learn_log_dir = args.learn_log_dir + f'u0={u_bound:.2f}_v0={v_bound:.2f}/'
    ckpt_path = args.id_log_dir + f'u0={u_bound:.2f}_v0={v_bound:.2f}/' + f'tau_{args.tau_s}/seed1/checkpoints/epoch-{args.id_epoch}.ckpt'
    workers = Learn_Slow_Fast(args, data_dir, ckpt_path, learn_log_dir, 'train')

    return workers


def get_local_ID_per_tau(args, u, v, u_min, v_min, u_step, v_step, id_heatmap, loss_tau):

    for tau in np.arange(args.tau_1, args.tau_N+args.dt, args.tau_unit):
        tau = round(tau, 2)
        u_index = int((u-u_min)/u_step)
        v_index = int((v-v_min)/v_step)
        tau_index = int((tau-args.tau_1)/args.tau_unit)
        
        with open(args.id_log_dir+f'u0={u:.2f}_v0={v:.2f}/'+f'tau_{tau}/test_log.txt', 'r') as log:
            id_heatmap[tau_index, u_index, v_index] = float(log.readlines()[-1].split(',')[-1])
        
        loss_f = np.load(args.id_log_dir+f'u0={u:.2f}_v0={v:.2f}'+f'/tau_{tau}/seed1/training_loss.npy')
        loss_tau[tau_index, u_index, v_index] = loss_f[-1,0]


def plot_ID_heatmap_per_tau(args, id_heatmap, loss_tau, u_step, u_min, u_max, v_step, v_min, v_max):

    os.makedirs(args.id_log_dir+'id_heatmap', exist_ok=True)
    T = np.arange(args.tau_1, args.tau_N+args.dt, args.tau_unit)
    for tau_index, tau in enumerate(T):
        tau = round(tau, 2)

        # ID heatmap
        plt.figure(figsize=(10,10))
        plt.title(f"tau={tau}", fontsize=18)
        # 热力图y轴由下到上
        # 画热力图
        plt.imshow(id_heatmap[tau_index])
        plt.tight_layout()
        # 色柱
        plt.clim(0, 2.5)
        plt.colorbar()
        # ID值写在对应区域中间
        for u_index in range(args.grid):
            for v_index in range(args.grid):
                plt.text(v_index, u_index, id_heatmap[tau_index, u_index, v_index].round(2), ha='center', va='center', fontsize=8)
        # 坐标轴
        plt.yticks(np.arange(0, args.grid, 1), (u_step/2+np.arange(u_min, u_max, u_step)).round(2), fontsize=12)
        plt.xticks(np.arange(0, args.grid, 1), (v_step/2+np.arange(v_min, v_max, v_step)).round(2), fontsize=12)
        plt.ylabel('u0 (fast)', fontsize=18)
        plt.xlabel('v0 (slow)', fontsize=18)
        # y轴由下到上
        plt.gca().invert_yaxis()
        plt.savefig(args.id_log_dir+'id_heatmap/'+f"tau_{tau}_id.jpg", dpi=300)
        plt.close()

        # loss 3D column chart
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"prior loss | tau={tau}", fontsize=18)
        ax.set_xlabel('u0 (fast)', fontsize=18)
        ax.set_ylabel('v0 (slow)', fontsize=18)
        ax.set_zlim(0, 1e-2)
        ax.set_xticks(np.arange(0, args.grid, 1))
        ax.set_yticks(np.arange(0, args.grid, 1))
        ax.set_xticklabels((u_step/2+np.arange(u_min, u_max, u_step)).round(2), fontsize=12)
        ax.set_yticklabels((v_step/2+np.arange(v_min, v_max, v_step)).round(2), fontsize=12)
        ax.set_zticks(np.arange(0, 1.1e-2, 2e-3))
        ax.set_zticklabels(np.arange(0, 1.1e-2, 2e-3).round(3), fontsize=12)
        ax.view_init(35, 20)
        plt.gca().invert_xaxis()
        # 画柱状图
        for u_index in range(args.grid):
            for v_index in range(args.grid):
                ax.bar3d(u_index, v_index, 0, 0.8, 0.8, loss_tau[tau_index, u_index, v_index], color='b', alpha=0.55, shade=True)
        plt.savefig(args.id_log_dir+'id_heatmap/'+f"tau_{tau}_loss.jpg", dpi=300)
        plt.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ours', help='Model: [ours, lstm, tcn, neural_ode]')
    parser.add_argument('--system', type=str, default='2S2F', help='Dynamical System: [1S1F, 1S2F, ToggleSwitch, SignalingCascade, HalfMoon, 2S2F, FHN, SC]')
    parser.add_argument('--channel_num', type=int, default=4, help='Overall featrue number')
    parser.add_argument('--obs_dim', type=int, default=4, help='Obervable feature dimension')
    parser.add_argument('--data_dim', type=int, default=4, help='Overall feature dimension')
    parser.add_argument('--trace_num', type=int, default=200, help='Number of simulation trajectories')
    parser.add_argument('--grid', type=int, default=5, help='Grid size of 2D data space') 
    parser.add_argument('--total_t', type=float, default=5.1, help='Time length of each simulation trajectories')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step of each simulation trajectories')
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

    if 'FHN' in args.system:
        from Data.generator_fhn import generate_dataset_static, generate_dataset_sliding_window
        from util.plot_fhn import plot_epoch_test_log, plot_id_per_tau
    else:
        raise NameError(f"System {args.system} not implemented!")

    if not args.parallel and args.cpu_num==1:
        print('Not recommand to limit the cpu num when non-parallellism!')

    # ctrl c
    def term_sig_handler(signum, frame):
        exit(0)
    signal.signal(signal.SIGINT, term_sig_handler)
    
    # main pipeline
    u_min, u_max = -2., 2.
    v_min, v_max = -0.5, 1.5
    u_step = (u_max-u_min)/args.grid
    v_step = (v_max-v_min)/args.grid
    u_bound = np.arange(u_min, u_max, u_step)
    v_bound = np.arange(v_min, v_max, v_step)
    u_bound, v_bound = np.meshgrid(u_bound, v_bound)
    id_heatmap = np.zeros((int((args.tau_N+args.dt-args.tau_1)/args.tau_unit)+1, args.grid, args.grid))
    loss_tau = np.zeros((int((args.tau_N+args.dt-args.tau_1)/args.tau_unit)+1, args.grid, args.grid))
    dynamic_learn_workers = []
    for i in range(args.grid):
        for j in range(args.grid):
            u = u_bound[i, j]
            v = v_bound[i, j] 
            print(f'\nu0 = {u:.2f}', f'v0 = {v:.2f}')
            # if round(v,2)>-0.25:
            #     continue
            local_ID_estimate(args, u, v, u_step, v_step)
            # if i%4==0 and j%4==0:  # 只取1/16的区域提取慢变量，避免子进程过多挤占资源
            #     dynamic_learn_workers.append(*local_Dynamic_Learn(args, 50, u, v, u_step, v_step))
            get_local_ID_per_tau(args, u, v, u_min, v_min, u_step, v_step, id_heatmap, loss_tau)
    plot_ID_heatmap_per_tau(args, id_heatmap, loss_tau, u_step, u_min, u_max, v_step, v_min, v_max)

    # block
    while any([sub.exitcode==None for sub in dynamic_learn_workers]):
        pass

    # plot local ID gif
    images1 = []
    images2 = []
    for tau in np.arange(args.tau_1, args.tau_N, args.tau_unit):
        tau = round(tau, 2)
        images1.append(imageio.imread(args.id_log_dir + f'id_heatmap/tau_{tau}_id.jpg'))
        images2.append(imageio.imread(args.id_log_dir + f'id_heatmap/tau_{tau}_loss.jpg'))
    imageio.mimsave(args.id_log_dir + f'id_heatmap/id.gif', images1, duration=0.5)
    imageio.mimsave(args.id_log_dir + f'id_heatmap/loss.gif', images2, duration=0.5)