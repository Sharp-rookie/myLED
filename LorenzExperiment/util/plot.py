import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_epoch_test_log(tau, max_epoch):

    class MSE():
        def __init__(self, tau):
            self.tau = tau
            self.each_mse = [[[] for _ in range(max_epoch)] for __ in range(20)]
            self.LB_id = [[] for _ in range(max_epoch)]
            self.MiND_id = [[] for _ in range(max_epoch)]
            self.MADA_id = [[] for _ in range(max_epoch)]
            self.PCA_id = [[] for _ in range(max_epoch)]

    fp = open(f'logs/time-lagged/tau_{tau}/test_log.txt', 'r')
    items = []
    for line in fp.readlines():
        tau = float(line[:-1].split(',')[0])
        seed = int(line[:-1].split(',')[1])
        mse = []
        for i in range(20):
            mse.append(float(line[:-1].split(',')[i+2]))
        epoch = int(line[:-1].split(',')[22])
        LB_id = float(line[:-1].split(',')[23])
        MiND_id = float(line[:-1].split(',')[24])
        MADA_id = float(line[:-1].split(',')[25])
        PCA_id = float(line[:-1].split(',')[26])

        find = False
        for M in items:
            if M.tau == tau:
                for i in range(20):
                    M.each_mse[i][epoch].append(mse[i])
                M.LB_id[epoch].append(LB_id)
                M.MiND_id[epoch].append(MiND_id)
                M.MADA_id[epoch].append(MADA_id)
                M.PCA_id[epoch].append(PCA_id)
                find = True
                    
        if not find:
            M = MSE(tau)
            for i in range(20):
                M.each_mse[i][epoch].append(mse[i])
            M.LB_id[epoch].append(LB_id)
            M.MiND_id[epoch].append(MiND_id)
            M.MADA_id[epoch].append(MADA_id)
            M.PCA_id[epoch].append(PCA_id)
            items.append(M)
    fp.close()

    for M in items:
        mse_list = [[] for _ in range(20)]
        LB_id_list = []
        MiND_id_list = []
        MADA_id_list = []
        PCA_id_list = []
        for epoch in range(max_epoch):
            for i in range(20):
                mse_list[i].append(np.mean(M.each_mse[i][epoch]))
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
    for i in range(20):
        plt.plot(range(max_epoch), mse_list[i], label=f'y{i+1}')
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
            epoch = int(line[:-1].split(',')[22])
            LB_id = float(line[:-1].split(',')[23])
            MiND_id = float(line[:-1].split(',')[24])
            MADA_id = float(line[:-1].split(',')[25])
            PCA_id = float(line[:-1].split(',')[26])

            if epoch in id_epoch:
                id_per_tau[i].append([LB_id, MiND_id, MADA_id, PCA_id])
    
    for i in range(len(tau_list)):
        id_per_tau[i] = np.mean(id_per_tau[i], axis=0)
    id_per_tau = np.array(id_per_tau)

    round_id_per_tau = []
    for id in id_per_tau:
        round_id_per_tau.append([round(id[0]),round(id[1]),round(id[2]),round(id[3])])
    round_id_per_tau = np.array(round_id_per_tau)

    import scienceplots
    plt.style.use(['science'])
    plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size':16})
    # for i, item in enumerate(['MLE','MiND','MADA','PCA']):
    for i, item in enumerate(['MLE']):
        plt.plot(tau_list, id_per_tau[:,i], marker="o", markersize=6, label="ID")
        plt.plot(tau_list, round_id_per_tau[:,i], marker="^", markersize=6, label="ID-rounding")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.xlabel(r'$\tau / s$', fontsize=18)
    plt.ylabel('Intrinsic dimensionality', fontsize=18)
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
    import scienceplots
    plt.style.use(['science'])
    plt.figure(figsize=(12,8))
    plt.plot(lag_list*5e-6, np.array(corrX), label='X')
    plt.plot(lag_list*5e-6, np.array(corrY), label='Y')
    plt.plot(lag_list*5e-6, np.array(corrZ), label='Z')
    plt.xlabel('time/s')
    plt.legend()
    plt.title('Autocorrelation')
    plt.savefig('corr.pdf', dpi=300)
    

def plot_pollu_autocorr():

    simdata = np.load('Data/origin/origin.npz')
    
    trace_num = 3
    corr = [[[] for _ in range(trace_num)] for _ in range(20)]
    for trace_id in range(trace_num):
        tmp = np.array(simdata['trace'])[trace_id]
        data = []
        for i in range(20):
            data.append(tmp[:,i][:,np.newaxis])

        data = pd.DataFrame(np.concatenate(data, axis=-1), columns=[f'y{i}' for i in range(20)])
        
        lag_list = np.arange(0, int(2.5*100), 1)
        for lag in tqdm(lag_list):
            for i in range(20):
                corr[i][trace_id].append(data[f'y{i}'].autocorr(lag=lag))
    
    corr = np.mean(corr, axis=1)

    import scienceplots
    plt.style.use(['science'])
    plt.figure(figsize=(20,16))
    for i in range(20):
        ax = plt.subplot(4,5,i+1)
        ax.plot(lag_list*1e-2, np.array(corr[i]))
        ax.set_xlabel(r'$t/s$')
        ax.set_ylabel(f'y{i}')
    # plt.legend()
    plt.subplots_adjust(wspace=0.45, hspace=0.45)
    plt.savefig('corr.pdf', dpi=300)
    
    
def plot_evolve(length):
    
    our = open(f'results/pretrain100_evolve_test_{length}.txt', 'r')
    lstm = open(f'results/lstm_evolve_test_{length}.txt', 'r')
    tcn = open(f'results/tcn_evolve_test_{length}.txt', 'r')
    ode = open(f'results/neuralODE_evolve_test_{length}.txt', 'r')
    
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
            c1_mae = float(line.split(',')[6])
            c2_mae = float(line.split(',')[7])
            
            if i==0:
                our_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae]),c1_mae,c2_mae])
            elif i==1:
                lstm_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae])])
            elif i==2:
                tcn_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae])])
            elif i==3:
                ode_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae])])
    
    our_data = np.mean(np.array(our_data), axis=0)
    lstm_data = np.mean(np.array(lstm_data), axis=0)
    tcn_data = np.mean(np.array(tcn_data), axis=0)
    ode_data = np.mean(np.array(ode_data), axis=0)

    import scienceplots
    plt.style.use(['science'])
    
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
    plt.savefig(f'results/evolve_test_{length}.pdf', dpi=300)

    for i, item in enumerate(['RMSE', 'MAPE']):
        plt.figure(figsize=(6,6))
        plt.rcParams.update({'font.size':16})
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
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(f'results/evolve_comp_{item}.pdf', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.rcParams.update({'font.size':16})
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
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    plt.xlabel(r'$t / s$', fontsize=18)
    plt.ylabel('MAE', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'results/slow_evolve_mae.pdf', dpi=300)

    plt.figure(figsize=(4,4))
    plt.rcParams.update({'font.size':16})
    plt.plot(our_data[:,0], our_data[:,3], marker="o", markersize=6, label=r'$overall$')
    plt.plot(our_data[:,0], our_data[:,6], marker="^", markersize=6, label=r'$c_1$')
    plt.plot(our_data[:,0], our_data[:,7], marker="D", markersize=6, label=r'$c_2$')
    plt.xlabel(r'$t / s$', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'results/our_slow_evolve_mae.pdf', dpi=300)
    
    item = ['our','lstm','tcn', 'ode', 'our old']
    for i, data in enumerate([our_data, lstm_data, tcn_data, ode_data]):
        print(f'{item[i]} | tau[{data[0,0]:.3f}] RMSE={data[0,2]:.4f}, MAE={data[0,3]:.4f}, MAPE={100*data[0,4]:.2f}% | tau[{data[9,0]:.3f}] RMSE={data[9,2]:.4f}, MAE={data[9,3]:.4f}, MAPE={100*data[9,4]:.2f}% | tau[{data[49,0]:.3f}] RMSE={data[49,2]:.4f}, MAE={data[49,3]:.4f}, MAPE={100*data[49,4]:.2f}%')
    

if __name__ == '__main__':
    
    plot_pollu_autocorr()
    # plot_evolve(0.8)