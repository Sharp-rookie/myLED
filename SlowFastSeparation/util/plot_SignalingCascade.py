import os
import math
import numpy as np
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})


def plot_epoch_test_log(tau, max_epoch, log_dir):

    class MSE():
        def __init__(self, tau):
            self.tau = tau
            self.mse_X1 = [[] for _ in range(max_epoch)]
            self.mse_X2 = [[] for _ in range(max_epoch)]
            self.mse_X3 = [[] for _ in range(max_epoch)]
            self.mse_X4 = [[] for _ in range(max_epoch)]
            self.LB_id = [[] for _ in range(max_epoch)]

    fp = open(log_dir + f'tau_{tau}/test_log.txt', 'r')
    items = []
    epoches = []
    for line in fp.readlines():
        tau = float(line[:-1].split(',')[0])
        seed = int(line[:-1].split(',')[1])
        mse_X1 = float(line[:-1].split(',')[2])
        mse_X2 = float(line[:-1].split(',')[3])
        mse_X3 = float(line[:-1].split(',')[4])
        mse_X4 = float(line[:-1].split(',')[5])
        epoch = int(line[:-1].split(',')[6])
        LB_id = float(line[:-1].split(',')[7])

        find = False
        for M in items:
            if M.tau == tau:
                M.mse_X1[epoch].append(mse_X1)
                M.mse_X2[epoch].append(mse_X2)
                M.mse_X3[epoch].append(mse_X3)
                M.LB_id[epoch].append(LB_id)
                find = True
                    
        if not find:
            M = MSE(tau)
            M.mse_X1[epoch].append(mse_X1)
            M.mse_X2[epoch].append(mse_X2)
            M.mse_X3[epoch].append(mse_X3)
            M.LB_id[epoch].append(LB_id)
            items.append(M)

        if epoch not in epoches:
            epoches.append(epoch)
    fp.close()

    for M in items:
        mse_X1_list = []
        mse_X2_list = []
        mse_X3_list = []
        mse_X4_list = []
        LB_id_list = []
        MiND_id_list = []
        MADA_id_list = []
        PCA_id_list = []
        for epoch in epoches:
            mse_X1_list.append(np.mean(M.mse_X1[epoch]))
            mse_X2_list.append(np.mean(M.mse_X2[epoch]))
            mse_X3_list.append(np.mean(M.mse_X3[epoch]))
            mse_X4_list.append(np.mean(M.mse_X4[epoch]))
            LB_id_list.append(np.mean(M.LB_id[epoch]))

    plt.figure(figsize=(12,9))
    plt.title(f'tau = {M.tau}')
    ax1 = plt.subplot(2,1,1)
    plt.xlabel('epoch')
    plt.ylabel('ID')
    plt.plot(epoches, LB_id_list)
    ax2 = plt.subplot(2,1,2)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.plot(epoches, mse_X1_list, label='X1')
    plt.plot(epoches, mse_X2_list, label='X2')
    plt.plot(epoches, mse_X3_list, label='X3')
    plt.plot(epoches, mse_X4_list, label='X4')
    # plt.ylim((0., 1.05*max(np.max(mse_x_list), np.max(mse_y_list), np.max(mse_z_list))))
    plt.legend()
    plt.savefig(log_dir + f'tau_{tau}/ID_per_epoch.pdf', dpi=300)
    plt.close()


def plot_id_per_tau(tau_list, id_epoch, log_dir):

    id_per_tau = [[] for _ in tau_list]
    for i, tau in enumerate(tau_list):
        fp = open(log_dir + f'tau_{round(tau,2)}/test_log.txt', 'r')
        for line in fp.readlines():
            seed = int(line[:-1].split(',')[1])
            epoch = int(line[:-1].split(',')[6])
            LB_id = float(line[:-1].split(',')[7])

            if epoch in id_epoch:
                id_per_tau[i].append([LB_id])
    
    for i in range(len(tau_list)):
        id_per_tau[i] = np.mean(id_per_tau[i], axis=0)
    id_per_tau = np.array(id_per_tau)

    round_id_per_tau = []
    for id in id_per_tau:
        if math.isnan(id[0]):
            id[0] = 0.
        round_id_per_tau.append([round(id[0])])
    round_id_per_tau = np.array(round_id_per_tau)

    plt.figure(figsize=(6,6))
    for i, item in enumerate(['MLE']):
        plt.plot(tau_list, id_per_tau[:,i], marker="o", markersize=6, label="ID")
        plt.plot(tau_list, round_id_per_tau[:,i], marker="^", markersize=6, label="ID-rounding")
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
        loss = np.load(f'logs/LearnDynamics/tau_{tau}/pretrain_epoch{pretrain_epoch}/delta_t{delta_t}/id{id}/val_loss_curve.npy')
        plt.plot(loss, label=f'ID[{id}]')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'Val mse | tau[{tau}] | pretrain_epoch[{pretrain_epoch}] | delta_t[{delta_t}]')
    plt.savefig(f'logs/LearnDynamics/tau_{tau}/pretrain_epoch{pretrain_epoch}/delta_t{delta_t}/val_loss_curves.pdf', dpi=300)


def plot_autocorr(T, dt=0.01):

    if os.path.exists('Data/1S2F/autocorr.pdf'): return

    data = np.load('Data/1S2F/origin/1/data.npz')
    X = np.array(data['X'])[:, np.newaxis]
    Y = np.array(data['Y'])[:, np.newaxis]
    Z = np.array(data['Z'])[:, np.newaxis]

    data = pd.DataFrame(np.concatenate((X,Y,Z), axis=-1), columns=['X', 'Y', 'Z'])
    
    # corr_matrix
    fig, ax = plt.subplots(figsize=(6,6))
    corr = data.corr()
    im = ax.imshow(corr, cmap='coolwarm')
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    ax.tick_params(axis='x', rotation=45)
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            text = ax.text(j, i, round(corr.iloc[i, j], 2),
                        ha='center', va='center', color='w')
    fig.colorbar(im)
    plt.savefig('Data/1S2F/corr.pdf', bbox_inches='tight')
    
    # auto corr series
    corrX, corrY, corrZ = [], [], []
    lag_list = np.arange(0, T*int(1/dt), int(1/dt))
    from tqdm import tqdm
    for lag in tqdm(lag_list):
        corrX.append(data['X'].autocorr(lag=lag))
        corrY.append(data['Y'].autocorr(lag=lag))
        corrZ.append(data['Z'].autocorr(lag=lag))

    plt.figure(figsize=(6,6))
    plt.plot(lag_list*dt, np.array(corrX), marker="o", markersize=6, label=r'$X$')
    plt.plot(lag_list*dt, np.array(corrY), marker="^", markersize=6, label=r'$Y$')
    plt.plot(lag_list*dt, np.array(corrZ), marker="D", markersize=6, label=r'$Z$')
    plt.xlabel(r'$t/s$', fontsize=18)
    plt.ylabel('Autocorrelation coefficient', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    # plt.subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig('Data/1S2F/autocorr.pdf', dpi=300)

    np.savez('Data/1S2F/autocorr.npz', corrX=corrX, corrY=corrY, corrZ=corrZ)


def plot_evolve(length):
    
    our = open(f'results/1S2F/evolve_test_{length}.txt', 'r')
    lstm = open(f'results/1S2F/lstm_evolve_test_{length}.txt', 'r')
    tcn = open(f'results/1S2F/tcn_evolve_test_{length/5}.txt', 'r')
    ode = open(f'results/1S2F/neuralODE_evolve_test_{length}.txt', 'r')
    
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
            
            if i==0:
                our_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==1:
                lstm_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==2:
                tcn_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==3:
                ode_data[seed-1].append([tau,mse,rmse,mae,mape])
    
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
        ax.plot(ode_data[:,0], ode_data[:,i+1], label='ode')
        ax.set_title(item)
        ax.set_xlabel('t / s')
        ax.legend()
    plt.savefig(f'results/1S2F/evolve_test_{length}.pdf', dpi=300)
    
    item = ['our','lstm','tcn', 'ode']
    for i, data in enumerate([our_data, lstm_data, tcn_data, ode_data]):
        print(f'{item[i]} | tau[{data[0,0]:.3f}] RMSE={data[0,2]:.4f}, MAE={data[0,3]:.4f}, MAPE={100*data[0,4]:.2f}% | tau[{data[9,0]:.3f}] RMSE={data[9,2]:.4f}, MAE={data[9,3]:.4f}, MAPE={100*data[9,4]:.2f}% | tau[{data[49,0]:.3f}] RMSE={data[49,2]:.4f}, MAE={data[49,3]:.4f}, MAPE={100*data[49,4]:.2f}%')


if __name__ == '__main__':
    
    plot_evolve(3.0)
