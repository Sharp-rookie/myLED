import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_epoch_test_log(tau, max_epoch):

    class MSE():
        def __init__(self, tau):
            self.tau = tau
            self.mse_c1 = [[] for _ in range(max_epoch)]
            self.mse_c2 = [[] for _ in range(max_epoch)]
            self.mse_c3 = [[] for _ in range(max_epoch)]
            self.mse_c4 = [[] for _ in range(max_epoch)]
            self.LB_id = [[] for _ in range(max_epoch)]
            self.MiND_id = [[] for _ in range(max_epoch)]
            self.MADA_id = [[] for _ in range(max_epoch)]
            self.PCA_id = [[] for _ in range(max_epoch)]

    fp = open(f'logs/time-lagged/tau_{tau}/test_log.txt', 'r')
    items = []
    for line in fp.readlines():
        tau = float(line[:-1].split(',')[0])
        seed = int(line[:-1].split(',')[1])
        mse_c1 = float(line[:-1].split(',')[2])
        mse_c2 = float(line[:-1].split(',')[3])
        mse_c3 = float(line[:-1].split(',')[4])
        mse_c4 = float(line[:-1].split(',')[5])
        epoch = int(line[:-1].split(',')[6])
        LB_id = float(line[:-1].split(',')[7])
        MiND_id = float(line[:-1].split(',')[8])
        MADA_id = float(line[:-1].split(',')[9])
        PCA_id = float(line[:-1].split(',')[10])

        find = False
        for M in items:
            if M.tau == tau:
                M.mse_c1[epoch].append(mse_c1)
                M.mse_c2[epoch].append(mse_c2)
                M.mse_c3[epoch].append(mse_c3)
                M.mse_c4[epoch].append(mse_c4)
                M.LB_id[epoch].append(LB_id)
                M.MiND_id[epoch].append(MiND_id)
                M.MADA_id[epoch].append(MADA_id)
                M.PCA_id[epoch].append(PCA_id)
                find = True
                    
        if not find:
            M = MSE(tau)
            M.mse_c1[epoch].append(mse_c1)
            M.mse_c2[epoch].append(mse_c2)
            M.mse_c3[epoch].append(mse_c3)
            M.mse_c4[epoch].append(mse_c4)
            M.LB_id[epoch].append(LB_id)
            M.MiND_id[epoch].append(MiND_id)
            M.MADA_id[epoch].append(MADA_id)
            M.PCA_id[epoch].append(PCA_id)
            items.append(M)
    fp.close()

    for M in items:
        mse_c1_list = []
        mse_c2_list = []
        mse_c3_list = []
        mse_c4_list = []
        LB_id_list = []
        MiND_id_list = []
        MADA_id_list = []
        PCA_id_list = []
        for epoch in range(max_epoch):
            mse_c1_list.append(np.mean(M.mse_c1[epoch]))
            mse_c2_list.append(np.mean(M.mse_c2[epoch]))
            mse_c3_list.append(np.mean(M.mse_c3[epoch]))
            mse_c4_list.append(np.mean(M.mse_c4[epoch]))
            LB_id_list.append(np.mean(M.LB_id[epoch]))
            MiND_id_list.append(np.mean(M.MiND_id[epoch]))
            MADA_id_list.append(np.mean(M.MADA_id[epoch]))
            PCA_id_list.append(np.mean(M.PCA_id[epoch]))

    plt.figure(figsize=(12,9))
    plt.title(f'tau = {M.tau}')
    ax1 = plt.subplot(2,1,1)
    plt.xlabel('epoch')
    plt.ylabel('ID')
    plt.plot(range(max_epoch), LB_id_list, label='LB')
    # plt.plot(range(max_epoch), MiND_id_list, label='MiND_ML')
    # plt.plot(range(max_epoch), MADA_id_list, label='MADA')
    # plt.plot(range(max_epoch), PCA_id_list, label='PCA')
    plt.legend()
    ax2 = plt.subplot(2,1,2)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.plot(range(max_epoch), mse_c1_list, label='c1')
    plt.plot(range(max_epoch), mse_c2_list, label='c2')
    plt.plot(range(max_epoch), mse_c3_list, label='c3')
    plt.plot(range(max_epoch), mse_c4_list, label='c4')
    # plt.ylim((0., 1.05*max(np.max(mse_c1_list), np.max(mse_c2_list), np.max(mse_c3_list))))
    plt.legend()
    plt.savefig(f'logs/time-lagged/tau_{tau}/ID_per_epoch.pdf', dpi=300)
    plt.close()


def plot_id_per_tau(tau_list, id_epoch):

    id_per_tau = [[] for _ in tau_list]
    for i, tau in enumerate(tau_list):
        fp = open(f'logs/time-lagged/tau_{round(tau,2)}/test_log.txt', 'r')
        for line in fp.readlines():
            seed = int(line[:-1].split(',')[1])
            epoch = int(line[:-1].split(',')[6])
            LB_id = float(line[:-1].split(',')[7])
            MiND_id = float(line[:-1].split(',')[8])
            MADA_id = float(line[:-1].split(',')[9])
            PCA_id = float(line[:-1].split(',')[10])

            if epoch in id_epoch:
                id_per_tau[i].append([LB_id, MiND_id, MADA_id, PCA_id])
    
    for i in range(len(tau_list)):
        id_per_tau[i] = np.mean(id_per_tau[i], axis=0)
    id_per_tau = np.array(id_per_tau)

    round_id_per_tau = []
    for id in id_per_tau:
        round_id_per_tau.append([round(id[0]),round(id[1]),round(id[2]),round(id[3])])
    round_id_per_tau = np.array(round_id_per_tau)

    plt.savefig('logs/time-lagged/id_per_tau.pdf', dpi=300)
    plt.figure()
    plt.rcParams.update({'font.size':15})
    # for i, item in enumerate(['MLE','MiND','MADA','PCA']):
    for i, item in enumerate(['MLE']):
        plt.plot(tau_list, id_per_tau[:,i], marker="o", markersize=6, label="ID")
        plt.plot(tau_list, round_id_per_tau[:,i], marker="^", markersize=6, label="ID-rounding")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.xlabel(r'$\tau / s$', fontsize=17)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('logs/time-lagged/id_per_tau.pdf', dpi=300)

        
def plot_slow_ae_loss(tau=0.0, pretrain_epoch=30, delta_t=0.01, id_list = [1,2,3,4]):
    
    plt.figure()
    for id in id_list:
        loss = np.load(f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/id{id}/val_loss_curve.npy')
        plt.plot(loss, label=f'ID[{id}]')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'Val mse | tau[{tau}] | pretrain_epoch[{pretrain_epoch}] | delta_t[{delta_t}]')
    plt.savefig(f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/val_loss_curves.pdf', dpi=300)


def plot_pnas_autocorr():

    data = np.load('Data/origin/1/data.npz')
    X = np.array(data['X'])[:, np.newaxis]
    Y = np.array(data['Y'])[:, np.newaxis]
    Z = np.array(data['Z'])[:, np.newaxis]

    data = pd.DataFrame(np.concatenate((X,Y,Z), axis=-1), columns=['X', 'Y', 'Z'])
    
    corrX, corrY, corrZ = [], [], []
    lag_list = np.arange(0, 600*2000, 2000)
    for lag in tqdm(lag_list):
        corrX.append(data['X'].autocorr(lag=lag))
        corrY.append(data['Y'].autocorr(lag=lag))
        corrZ.append(data['Z'].autocorr(lag=lag))
    plt.figure(figsize=(12,8))
    plt.plot(lag_list*5e-6, np.array(corrX), label='X')
    plt.plot(lag_list*5e-6, np.array(corrY), label='Y')
    plt.plot(lag_list*5e-6, np.array(corrZ), label='Z')
    plt.xlabel('time/s')
    plt.legend()
    plt.title('Autocorrelation')
    plt.savefig('corr.pdf', dpi=300)
    

def plot_jcp12_autocorr():

    simdata = np.load('Data/origin/origin.npz')
    
    trace_num = 3
    corrC1, corrC2, corrC3, corrC4 = [[] for _ in range(trace_num)], [[] for _ in range(trace_num)], [[] for _ in range(trace_num)], [[] for _ in range(trace_num)]
    for trace_id in range(trace_num):
        tmp = np.array(simdata['trace'])[trace_id]
        c1 = tmp[:,0][:,np.newaxis]
        c2 = tmp[:,1][:,np.newaxis]
        c3 = tmp[:,2][:,np.newaxis]
        c4 = tmp[:,3][:,np.newaxis]

        data = pd.DataFrame(np.concatenate((c1,c2,c3,c4), axis=-1), columns=['c1','c2','c3','c4'])
        
        lag_list = np.arange(0, 5*100, 10)
        for lag in tqdm(lag_list):
            corrC1[trace_id].append(data['c1'].autocorr(lag=lag))
            corrC2[trace_id].append(data['c2'].autocorr(lag=lag))
            corrC3[trace_id].append(data['c3'].autocorr(lag=lag))
            corrC4[trace_id].append(data['c4'].autocorr(lag=lag))
    
    corrC1 = np.mean(corrC1, axis=0)
    corrC2 = np.mean(corrC2, axis=0)
    corrC3 = np.mean(corrC3, axis=0)
    corrC4 = np.mean(corrC4, axis=0)
    
    plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size':15})
    plt.plot(lag_list*1e-2, np.array(corrC1), marker="o", markersize=6, label=r'$c_1$')
    plt.plot(lag_list*1e-2, np.array(corrC2), marker="^", markersize=6, label=r'$c_2$')
    plt.plot(lag_list*1e-2, np.array(corrC3), marker="D", markersize=6, label=r'$c_3$')
    plt.plot(lag_list*1e-2, np.array(corrC4), marker="*", markersize=6, label=r'$c_4$')
    plt.xlabel(r'$t/s$', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('corr.pdf', dpi=300)
    
    
def plot_evolve(length):
    
    our = open(f'pretrain100_evolve_test_{length}.txt', 'r')
    lstm = open(f'lstm_evolve_test_{length}.txt', 'r')
    tcn = open(f'tcn_evolve_test_{length}.txt', 'r')
    
    our_data = [[] for seed in range(10)]
    lstm_data = [[] for seed in range(10)]
    tcn_data = [[] for seed in range(10)]
    for i, data in enumerate([our, lstm, tcn]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            mse = float(line.split(',')[2])
            rmse = float(line.split(',')[3])
            mae = float(line.split(',')[4])
            mape = float(line.split(',')[5])
            c1_mae = float(line.split(',')[6])
            c2_mae = float(line.split(',')[7])
            
            if i==0:
                our_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae]),c1_mae,c2_mae])
            elif i==1:
                lstm_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae])])
            elif i==2:
                tcn_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae])])
    
    our_data = np.mean(np.array(our_data), axis=0)
    lstm_data = np.mean(np.array(lstm_data), axis=0)
    tcn_data = np.mean(np.array(tcn_data), axis=0)
    
    plt.figure(figsize=(16,16))
    for i, item in enumerate(['mse', 'rmse', 'mae', 'mape']):
        ax = plt.subplot(2,2,i+1)
        ax.plot(our_data[:,0], our_data[:,i+1], label='our')
        ax.plot(lstm_data[:,0], lstm_data[:,i+1], label='lstm')
        ax.plot(tcn_data[:,0], tcn_data[:,i+1], label='tcn')
        ax.set_title(item)
        ax.set_xlabel('t / s')
        ax.legend()
    plt.savefig(f'evolve_test_{length}.pdf', dpi=300)

    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size':15})
    plt.plot(our_data[:,0], our_data[:,5], marker="o", markersize=6, label='Our Model')
    plt.plot(lstm_data[:,0], lstm_data[:,5], marker="^", markersize=6, label='LSTM')
    plt.plot(tcn_data[:,0], tcn_data[:,5], marker="D", markersize=6, label='TCN')
    plt.xlabel(r'$t / s$', fontsize=17)
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'slow_evolve_mae.pdf', dpi=300)

    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size':15})
    plt.plot(our_data[:,0], our_data[:,3], marker="o", markersize=6, label=r'$overall$')
    plt.plot(our_data[:,0], our_data[:,6], marker="^", markersize=6, label=r'$c_1$')
    plt.plot(our_data[:,0], our_data[:,7], marker="D", markersize=6, label=r'$c_2$')
    plt.xlabel(r'$t / s$', fontsize=17)
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'our_slow_evolve_mae.pdf', dpi=300)
    
    item = ['our','lstm','tcn']
    for i, data in enumerate([our_data, lstm_data, tcn_data]):
        print(f'{item[i]} | tau[{data[0,0]:.3f}] RMSE={data[0,2]:.4f}, MAE={data[0,3]:.4f}, MAPE={100*data[0,4]:.2f}% | tau[{data[9,0]:.3f}] RMSE={data[9,2]:.4f}, MAE={data[9,3]:.4f}, MAPE={100*data[9,4]:.2f}% | tau[{data[49,0]:.3f}] RMSE={data[49,2]:.4f}, MAE={data[49,3]:.4f}, MAPE={100*data[49,4]:.2f}%')
    

if __name__ == '__main__':
    
    # plot_jcp12_autocorr()
    plot_evolve(0.8)