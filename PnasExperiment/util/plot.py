import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_epoch_test_log(tau, max_epoch):

    class MSE():
        def __init__(self, tau):
            self.tau = tau
            self.mse_x = [[] for _ in range(max_epoch)]
            self.mse_y = [[] for _ in range(max_epoch)]
            self.mse_z = [[] for _ in range(max_epoch)]
            self.LB_id = [[] for _ in range(max_epoch)]
            self.MiND_id = [[] for _ in range(max_epoch)]
            self.MADA_id = [[] for _ in range(max_epoch)]
            self.PCA_id = [[] for _ in range(max_epoch)]

    fp = open(f'logs/time-lagged/tau_{tau}/test/log.txt', 'r')
    items = []
    for line in fp.readlines():
        tau = float(line[:-1].split(',')[0])
        seed = int(line[:-1].split(',')[1])
        mse_x = float(line[:-1].split(',')[2])
        mse_y = float(line[:-1].split(',')[3])
        mse_z = float(line[:-1].split(',')[4])
        epoch = int(line[:-1].split(',')[5])
        LB_id = float(line[:-1].split(',')[6])
        MiND_id = float(line[:-1].split(',')[7])
        MADA_id = float(line[:-1].split(',')[8])
        PCA_id = float(line[:-1].split(',')[9])

        find = False
        for M in items:
            if M.tau == tau:
                M.mse_x[epoch].append(mse_x)
                M.mse_y[epoch].append(mse_y)
                M.mse_z[epoch].append(mse_z)
                M.LB_id[epoch].append(LB_id)
                M.MiND_id[epoch].append(MiND_id)
                M.MADA_id[epoch].append(MADA_id)
                M.PCA_id[epoch].append(PCA_id)
                find = True
                    
        if not find:
            M = MSE(tau)
            M.mse_x[epoch].append(mse_x)
            M.mse_y[epoch].append(mse_y)
            M.mse_z[epoch].append(mse_z)
            M.LB_id[epoch].append(LB_id)
            M.MiND_id[epoch].append(MiND_id)
            M.MADA_id[epoch].append(MADA_id)
            M.PCA_id[epoch].append(PCA_id)
            items.append(M)
    fp.close()

    for M in items:
        mse_x_list = []
        mse_y_list = []
        mse_z_list = []
        LB_id_list = []
        MiND_id_list = []
        MADA_id_list = []
        PCA_id_list = []
        for epoch in range(max_epoch):
            mse_x_list.append(np.mean(M.mse_x[epoch]))
            mse_y_list.append(np.mean(M.mse_y[epoch]))
            mse_z_list.append(np.mean(M.mse_z[epoch]))
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
        plt.plot(range(max_epoch), MiND_id_list, label='MiND_ML')
        plt.plot(range(max_epoch), MADA_id_list, label='MADA')
        plt.plot(range(max_epoch), PCA_id_list, label='PCA')
        plt.legend()
        ax2 = plt.subplot(2,1,2)
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.plot(range(max_epoch), mse_x_list, label='x')
        plt.plot(range(max_epoch), mse_y_list, label='y')
        plt.plot(range(max_epoch), mse_z_list, label='z')
        # plt.ylim((0., 1.05*max(np.max(mse_x_list), np.max(mse_y_list), np.max(mse_z_list))))
        plt.legend()
        plt.savefig(f'logs/time-lagged/tau_{tau}/ID_per_epoch.jpg', dpi=300)
        plt.close()


def plot_id_per_tau(tau_list, id_epoch):

    id_per_tau = [[] for _ in range(tau_list)]
    for i, tau in enumerate(tau_list):
        fp = open(f'logs/time-lagged/tau_{tau}/test/log.txt', 'r')
        for line in fp.readlines():
            seed = int(line[:-1].split(',')[1])
            epoch = int(line[:-1].split(',')[5])
            LB_id = float(line[:-1].split(',')[6])
            MiND_id = float(line[:-1].split(',')[7])
            MADA_id = float(line[:-1].split(',')[8])
            PCA_id = float(line[:-1].split(',')[9])

            if epoch == id_epoch:
                id_per_tau[i].append([LB_id, MiND_id, MADA_id, PCA_id])
    
    id_per_tau = np.mean(id_per_tau, axis=-2)

    plt.figure()
    for i, item in enumerate(['MLE','MiND','MADA','PCA']):
        plt.plot(tau_list, id_per_tau[:,i], label=item)
    plt.legend()
    plt.xlabel('tau / s')
    plt.savefig('logs/time-lagged/id_per_tau.pdf', dpi=300)

        
def plot_slow_ae_loss(tau=0.0, pretrain_epoch=30, delta_t=0.01, id_list = [1,2,3,4]):
    
    plt.figure()
    for id in id_list:
        loss = np.load(f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/delta_t{delta_t}/id{id}/val_loss_curve.npy')
        plt.plot(loss, label=f'ID[{id}]')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'Val mse | tau[{tau}] | pretrain_epoch[{pretrain_epoch}] | delta_t[{delta_t}]')
    plt.savefig(f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/delta_t{delta_t}/val_loss_curves.jpg', dpi=300)


def plot_pnas_autocorr():

    data = np.load('Data/origin/1/data.npz')
    X = np.array(data['X'])[:, np.newaxis]
    Y = np.array(data['Y'])[:, np.newaxis]
    Z = np.array(data['Z'])[:, np.newaxis]

    data = pd.DataFrame(np.concatenate((X,Y,Z), axis=-1), columns=['X', 'Y', 'Z'])
    
    corrX, corrY, corrZ = [], [], []
    lag_list = np.arange(0, 600*2000, 2000)
    from tqdm import tqdm
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
    plt.savefig('corr.jpg', dpi=300)


def plot_evolve(tau):
    
    our = open(f'evolve_test_{tau}.txt', 'r')
    lstm = open(f'lstm_evolve_test_{tau}.txt', 'r')
    tcn = open(f'tcn_evolve_test_{tau}.txt', 'r')
    
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
            
            if i==0:
                our_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==1:
                lstm_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==2:
                tcn_data[seed-1].append([tau,mse,rmse,mae,mape])
    
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
    plt.savefig(f'evolve_test_{tau}.jpg', dpi=300)
    
    print(f'our | tau[{our_data[0,0]:.3f}] RMSE={our_data[0,2]:.4f}, MAPE={100*our_data[0,4]:.2f}% | tau[{our_data[9,0]:.3f}] RMSE={our_data[9,2]:.4f}, MAPE={100*our_data[9,4]:.2f}% | tau[{our_data[49,0]:.3f}] RMSE={our_data[49,2]:.4f}, MAPE={100*our_data[49,4]:.2f}%')
    print(f'lstm | tau[{lstm_data[0,0]:.3f}] RMSE={lstm_data[0,2]:.4f}, MAPE={100*lstm_data[0,4]:.2f}% | tau[{lstm_data[9,0]:.3f}] RMSE={lstm_data[9,2]:.4f}, MAPE={100*lstm_data[9,4]:.2f}% | tau[{lstm_data[49,0]:.3f}] RMSE={lstm_data[49,2]:.4f}, MAPE={100*lstm_data[49,4]:.2f}%')
    print(f'tcn | tau[{tcn_data[0,0]}:.3f] RMSE={tcn_data[0,2]:.4f}, MAPE={100*tcn_data[0,4]:.2f}% | tau[{tcn_data[9,0]:.3f}] RMSE={tcn_data[9,2]:.4f}, MAPE={100*tcn_data[9,4]:.2f}% | tau[{tcn_data[49,0]:.3f}] RMSE={tcn_data[49,2]:.4f}, MAPE={100*tcn_data[49,4]:.2f}%')


if __name__ == '__main__':
    
    # plot_pnas_autocorr()
    plot_evolve(2.5)
