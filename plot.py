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

        plt.figure(figsize=(8,8))
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
        
        
def plot_slow_ae_loss(tau=0.0, pretrain_epoch=[1], id_list = [1,2,3,4]):
    
    for epoch in pretrain_epoch:
        plt.figure()
        for id in id_list:
            loss = np.load(f'logs/slow_vars_koopman/tau_{tau}/pretrain_epoch{epoch}/id{id}/loss_curve.npy')
            plt.plot(loss, label=f'ID[{id}]')
        plt.xlabel('epoch')
        plt.legend()
        plt.title(f'tau[{tau}] | pretrain_epoch[{epoch}]')
        plt.savefig(f'logs/slow_vars_koopman/tau_{tau}/pretrain_epoch{epoch}/loss_curves.jpg', dpi=300)


def plot_y_corr():

    data = np.load('Data/origin/1/data.npz')
    X = np.array(data['X'])[:, np.newaxis]
    Y = np.array(data['Y'])[:, np.newaxis]
    Z = np.array(data['Z'])[:, np.newaxis]

    data = pd.DataFrame(np.concatenate((X,Y,Z), axis=-1), columns=['X', 'Y', 'Z'])
    
    corrX, corrY, corrZ = [], [], []
    lag_list = np.array(range(2000))
    for lag in lag_list:
        corrX.append(data['X'].autocorr(lag=lag))
        corrY.append(data['Y'].autocorr(lag=lag))
        corrZ.append(data['Z'].autocorr(lag=lag))
    plt.figure()
    plt.plot(lag_list*0.005, np.array(corrX), label='X')
    plt.plot(lag_list*0.005, np.array(corrY), label='Y')
    plt.plot(lag_list*0.005, np.array(corrZ), label='Z')
    plt.xlabel('time/s')
    plt.legend()
    plt.title('Autocorrelation')
    plt.savefig('corr.jpg', dpi=300)


if __name__ == '__main__':
    
    # [plot_epoch_test_log(round(tau, 3), max_epoch=50+1) for tau in np.arange(0., 2.51, 0.25)]
    [plot_epoch_test_log(round(tau, 5), max_epoch=50+1) for tau in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 3.0]]
    # plot_y_corr()
    # plot_slow_ae_loss(tau=1.5, pretrain_epoch=[2, 30], id_list=[1,2,3,4])