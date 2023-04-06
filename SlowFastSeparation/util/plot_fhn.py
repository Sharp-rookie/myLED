import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})


def plot_epoch_test_log(tau, max_epoch, log_dir):

    class MSE():
        def __init__(self, tau):
            self.tau = tau
            self.mse_u = [[] for _ in range(max_epoch)]
            self.mse_v = [[] for _ in range(max_epoch)]
            self.MLE_id = [[] for _ in range(max_epoch)]

    fp = open(log_dir + f'tau_{tau}/test_log.txt', 'r')
    items = []
    for line in fp.readlines():
        tau = float(line[:-1].split(',')[0])
        seed = int(line[:-1].split(',')[1])
        mse_u = float(line[:-1].split(',')[2])
        mse_v = float(line[:-1].split(',')[3])
        epoch = int(line[:-1].split(',')[4])
        MLE_id = float(line[:-1].split(',')[5])

        find = False
        for M in items:
            if M.tau == tau:
                M.mse_u[epoch].append(mse_u)
                M.mse_v[epoch].append(mse_v)
                M.MLE_id[epoch].append(MLE_id)
                find = True
                    
        if not find:
            M = MSE(tau)
            M.mse_u[epoch].append(mse_u)
            M.mse_v[epoch].append(mse_v)
            M.MLE_id[epoch].append(MLE_id)
            items.append(M)
    fp.close()

    for M in items:
        mse_u_list = []
        mse_v_list = []
        MLE_id_list = []
        for epoch in range(max_epoch):
            mse_u_list.append(np.mean(M.mse_u[epoch]))
            mse_v_list.append(np.mean(M.mse_v[epoch]))
            MLE_id_list.append(np.mean(M.MLE_id[epoch]))

    plt.figure(figsize=(12,9))
    plt.title(f'tau = {M.tau}')
    ax1 = plt.subplot(2,1,1)
    plt.xlabel('epoch')
    plt.ylabel('ID')
    plt.plot(range(max_epoch), MLE_id_list)
    ax2 = plt.subplot(2,1,2)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.plot(range(max_epoch), mse_u_list, label='u')
    plt.plot(range(max_epoch), mse_v_list, label='v')
    # plt.ylim((0., 1.05*max(np.max(mse_v_list), np.max(mse_w_list), np.max(mse_b_list))))
    plt.legend()
    plt.savefig(log_dir + f'tau_{tau}/ID_per_epoch.pdf', dpi=300)
    plt.close()


def plot_id_per_tau(tau_list, id_epoch, log_dir):

    id_per_tau = [[] for _ in tau_list]
    for i, tau in enumerate(tau_list):
        fp = open(log_dir + f'tau_{round(tau,2)}/test_log.txt', 'r')
        for line in fp.readlines():
            seed = int(line[:-1].split(',')[1])
            epoch = int(line[:-1].split(',')[4])
            MLE_id = float(line[:-1].split(',')[5])

            if epoch in id_epoch:
                if MLE_id == float('inf'):
                    MLE_id = 0.
                elif math.isnan(MLE_id):
                    MLE_id = 0.
                id_per_tau[i].append([MLE_id])
    
    for i in range(len(tau_list)):
        if len(id_per_tau[i])==0:
            id_per_tau[i].append([0.])
        id_per_tau[i] = np.mean(id_per_tau[i], axis=0)
    id_per_tau = np.array(id_per_tau)

    round_id_per_tau = []
    for id in id_per_tau:
        round_id_per_tau.append([round(np.mean(id))])
    round_id_per_tau = np.array(round_id_per_tau)

    plt.figure(figsize=(6,6))
    plt.plot(tau_list, id_per_tau[:,0], marker="o", markersize=6, label="ID")
    plt.plot(tau_list, round_id_per_tau[:,0], marker="^", markersize=6, label="ID-rounding")
    plt.ylim((0, 3))  # assume the max ID is 3
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.xlabel(r'$\tau / s$', fontsize=18)
    plt.ylabel('Intrinsic dimensionality', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(log_dir + 'id_per_tau.pdf', dpi=300)

        
def plot_slow_ae_loss(tau=0.0, pretrain_epoch=30, delta_t=0.01, id_list = [1,2,3,4]):
    
    plt.figure()
    for id in id_list:
        loss = np.load(f'logs/LearnDynamics/tau_{tau}/pretrain_epoch{pretrain_epoch}/id{id}/val_loss_curve.npy')
        plt.plot(loss, label=f'ID[{id}]')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'Val mse | tau[{tau}] | pretrain_epoch[{pretrain_epoch}] | delta_t[{delta_t}]')
    plt.savefig(f'logs/LearnDynamics/tau_{tau}/pretrain_epoch{pretrain_epoch}/val_loss_curves.pdf', dpi=300)
    

def plot_autocorr(T, dt=0.01):

    # if os.path.exists('Data/FHN/autocorr.pdf'): return

    simdata = np.load('result.npz')
    
    trace_num = 10
    corrU, corrV = [[] for _ in range(trace_num)], [[] for _ in range(trace_num)]
    for trace_id in range(trace_num):
        tmp = np.array(simdata['results'])[trace_id]
        u = tmp[:,0][:,np.newaxis]
        v = tmp[:,1][:,np.newaxis]

        data = pd.DataFrame(np.concatenate((u,v), axis=-1), columns=['u','v'])
        
        fig, ax = plt.subplots(figsize=(6,6))
        corr = data.corr()
        im = ax.imshow(corr, cmap='coolwarm')
        ax.set_xticks(range(len(data.columns)), fontsize=16)
        ax.set_yticks(range(len(data.columns)), fontsize=16)
        ax.set_xticklabels(data.columns, fontsize=18)
        ax.set_yticklabels(data.columns,fontsize=18)
        ax.tick_params(axis='x', rotation=45)
        for i in range(len(data.columns)):
            for j in range(len(data.columns)):
                text = ax.text(j, i, round(corr.iloc[i, j], 2),
                            ha='center', va='center', color='w')
        fig.colorbar(im)
        plt.savefig('corr.pdf', bbox_inches='tight')

        lag_list = np.arange(0, T*int(1/dt), int(1/dt))
        for lag in tqdm(lag_list):
            corrU[trace_id].append(data['u'].autocorr(lag=lag))
            corrV[trace_id].append(data['v'].autocorr(lag=lag))
    
    corrU = np.mean(corrU, axis=0)
    corrV = np.mean(corrV, axis=0)

    plt.figure(figsize=(6,6))
    plt.plot(lag_list*dt, np.array(corrU), marker="o", markersize=6, label=r'$v$')
    plt.plot(lag_list*dt, np.array(corrV), marker="^", markersize=6, label=r'$w$')
    plt.xlabel(r'$t/s$', fontsize=18)
    plt.ylabel('Autocorrelation coefficient', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    # plt.subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig('autocorr.pdf', dpi=300)

    # np.savez('autocorr.npz', corrV=corrU, corrW=corrV)
    
    
def plot_evolve(length):
    
    our = open(f'results/FHN/pretrain100_evolve_test_{length}.txt', 'r')
    lstm = open(f'results/FHN/lstm_evolve_test_{length}.txt', 'r')
    tcn = open(f'results/FHN/tcn_evolve_test_{length}.txt', 'r')
    ode = open(f'results/FHN/neuralODE_evolve_test_{length}.txt', 'r')
    
    our_data = [[] for seed in range(10)]
    lstm_data = [[] for seed in range(10)]
    tcn_data = [[] for seed in range(10)]
    ode_data = [[] for seed in range(10)]
    for i, data in enumerate([our, lstm, tcn, ode]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            mse = float(line.split(',')[2])
            rmse = float(line.split(',')[3])
            mae = float(line.split(',')[4])
            mape = float(line.split(',')[5])
            v_mae = float(line.split(',')[6])
            w_mae = float(line.split(',')[7])
            
            if i==0:
                our_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([v_mae,w_mae]),v_mae,w_mae])
            elif i==1:
                lstm_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([v_mae,w_mae])])
            elif i==2:
                tcn_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([v_mae,w_mae])])
            elif i==3:
                ode_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([v_mae,w_mae])])
    
    our_data = np.mean(np.array(our_data), axis=0)
    lstm_data = np.mean(np.array(lstm_data), axis=0)
    tcn_data = np.mean(np.array(tcn_data), axis=0)
    ode_data = np.mean(np.array(ode_data), axis=0)
    
    plt.figure(figsize=(16,16))
    for i, item in enumerate(['mse', 'rmse', 'mae', 'mape']):
        ax = plt.subplot(2,2,i+1)
        ax.plot(our_data[:,0], our_data[:,i+1], label='our')
        ax.plot(lstm_data[:,0], lstm_data[:,i+1], label='lstm')
        ax.plot(tcn_data[:,0], tcn_data[:,i+1], label='tcn')
        # ax.plot(ode_data[:,0], ode_data[:,i+1], label='ode')
        ax.set_title(item)
        ax.set_xlabel('t / s')
        ax.legend()
    plt.savefig(f'results/FHN/evolve_test_{length}.pdf', dpi=300)

    for i, item in enumerate(['RMSE', 'MAPE']):
        plt.figure(figsize=(6,6))
        ax = plt.subplot(1,1,1)
        ax.plot(our_data[::2,0], our_data[::2,2*(i+1)], marker="o", markersize=6, label='our')
        ax.plot(lstm_data[::2,0], lstm_data[::2,2*(i+1)], marker="^", markersize=6, label='lstm')
        ax.plot(tcn_data[::2,0], tcn_data[::2,2*(i+1)], marker="D", markersize=6, label='tcn')
        ax.plot(ode_data[::2,0], ode_data[::2,2*(i+1)], marker="+", markersize=6, label='ode')
        ax.set_xlabel(r'$t / s$', fontsize=18)
        ax.set_ylabel(item, fontsize=18)
        ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.1))
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = ax.inset_axes((0.16, 0.55, 0.47, 0.35))
        axins.plot(our_data[:int(len(our_data)*0.1):1,0], our_data[:int(len(our_data)*0.1):1,2*(i+1)], marker="o", markersize=6, label='our')
        axins.plot(lstm_data[:int(len(lstm_data)*0.1):1,0], lstm_data[:int(len(lstm_data)*0.1):1,2*(i+1)], marker="^", markersize=6, label='lstm')
        axins.plot(tcn_data[:int(len(tcn_data)*0.1):1,0], tcn_data[:int(len(tcn_data)*0.1):1,2*(i+1)], marker="D", markersize=6, label='tcn')
        # axins.plot(ode_data[:int(len(ode_data)*0.1):1,0], ode_data[:int(len(ode_data)*0.1):1,2*(i+1)], marker="+", markersize=6, label='ode')
        mark_inset(ax, axins, lov=3, low=1, fc="none", ec='k', lw=1)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(f'results/FHN/evolve_comp_{item}.pdf', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(our_data[::2,0], our_data[::2,5], marker="o", markersize=6, label='Our Model')
    ax.plot(lstm_data[::2,0], lstm_data[::2,5], marker="^", markersize=6, label='LSTM')
    # ax.plot(tcn_data[::2,0], tcn_data[::2,5], marker="D", markersize=6, label='TCN')
    ax.plot(ode_data[::2,0], ode_data[::2,5], marker="+", markersize=6, label='Neural ODE')
    ax.legend()
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = ax.inset_axes((0.13, 0.43, 0.43, 0.3))
    axins.plot(our_data[:int(len(our_data)*0.6):3,0], our_data[:int(len(our_data)*0.6):3,5], marker="o", markersize=6, label='Our Model')
    axins.plot(lstm_data[:int(len(lstm_data)*0.6):3,0], lstm_data[:int(len(lstm_data)*0.6):3,5], marker="^", markersize=6, label='LSTM')
    mark_inset(ax, axins, lov=3, low=1, fc="none", ec='k', lw=1)

    plt.xlabel(r'$t / s$', fontsize=18)
    plt.ylabel('MAE', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'results/FHN/slow_evolve_mae.pdf', dpi=300)

    plt.figure(figsize=(4,4))
    plt.plot(our_data[:,0], our_data[:,3], marker="o", markersize=6, label=r'$overall$')
    plt.plot(our_data[:,0], our_data[:,6], marker="^", markersize=6, label=r'$c_1$')
    plt.plot(our_data[:,0], our_data[:,7], marker="D", markersize=6, label=r'$c_2$')
    plt.xlabel(r'$t / s$', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'results/FHN/our_slow_evolve_mae.pdf', dpi=300)
    
    item = ['our','lstm','tcn', 'ode']
    for i, data in enumerate([our_data, lstm_data, tcn_data, ode_data]):
        print(f'{item[i]} | tau[{data[0,0]:.3f}] RMSE={data[0,2]:.4f}, MAE={data[0,3]:.4f}, MAPE={100*data[0,4]:.2f}% | tau[{data[9,0]:.3f}] RMSE={data[9,2]:.4f}, MAE={data[9,3]:.4f}, MAPE={100*data[9,4]:.2f}% | tau[{data[49,0]:.3f}] RMSE={data[49,2]:.4f}, MAE={data[49,3]:.4f}, MAPE={100*data[49,4]:.2f}%')
    

if __name__ == '__main__':
    
    # plot_autocorr(T=300)
    
    trace = np.load('results.npz')['trace']
    u = trace[0,:,0]
    v = trace[0,:,1]
    # derivative
    u_diff = np.gradient(u, 0.001)
    v_diff = np.gradient(v, 0.001)
    # transform
    u_weight = 2 / (np.exp(u_diff) + np.exp(-u_diff))
    v_weight = 2 / (np.exp(v_diff) + np.exp(-v_diff))
    # t
    dt = 0.001
    point = int(40/dt)
    t = np.arange(0, len(u_diff)) * dt
    # plot
    plt.figure(figsize=(12,5))
    ax = plt.subplot(1,2,1)
    # ax.plot(t[:point], u[:point], label='u')
    # ax.plot(t[:point], u_diff[:point], label='du/dt')
    # ax.plot(t[point:], u[point:], label='u')
    # ax.plot(t[point:], u_diff[point:], label='du/dt')
    # ax.plot(t[point:], u_weight[point:], label='u weight')
    ax.plot(t, u, label='u')
    ax.plot(t, u_diff, label='du/dt')
    ax.plot(t, u_weight, label='u weight')
    ax.set_xlabel('t/s', fontsize=18)
    ax.legend()
    ax.set_ylim(-2,2)
    ax = plt.subplot(1,2,2)
    # ax.plot(t[:point], v[:point], label='v')
    # ax.plot(t[:point], v_diff[:point], label='dv/dt')
    # ax.plot(t[point:], v[point:], label='v')
    # ax.plot(t[point:], v_diff[point:], label='dv/dt')
    # ax.plot(t[point:], v_weight[point:], label='v weight')
    ax.plot(t, v, label='v')
    ax.plot(t, v_diff, label='dv/dt')
    ax.plot(t, v_weight, label='v weight')
    ax.set_xlabel('t/s', fontsize=18)
    ax.legend()
    ax.set_ylim(-2,2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'results.jpg', dpi=300)
