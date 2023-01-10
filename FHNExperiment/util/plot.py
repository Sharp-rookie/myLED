from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_epoch_test_log(tau, max_epoch):

    class MSE():
        def __init__(self, tau):
            self.tau = tau
            self.mse_act = [[] for _ in range(max_epoch)]
            self.mse_in = [[] for _ in range(max_epoch)]
            self.LB_id = [[] for _ in range(max_epoch)]
            self.MiND_id = [[] for _ in range(max_epoch)]
            self.MADA_id = [[] for _ in range(max_epoch)]
            self.PCA_id = [[] for _ in range(max_epoch)]

    fp = open(f'logs/time-lagged/tau_{tau}/test/log.txt', 'r')
    items = []
    for line in fp.readlines():
        tau = float(line[:-1].split(',')[0])
        seed = int(line[:-1].split(',')[1])
        mse_act = float(line[:-1].split(',')[2])
        mse_in = float(line[:-1].split(',')[3])
        epoch = int(line[:-1].split(',')[4])
        LB_id = float(line[:-1].split(',')[5])
        MiND_id = float(line[:-1].split(',')[6])
        MADA_id = float(line[:-1].split(',')[7])
        PCA_id = float(line[:-1].split(',')[8])

        find = False
        for M in items:
            if M.tau == tau:
                M.mse_act[epoch].append(mse_act)
                M.mse_in[epoch].append(mse_in)
                M.LB_id[epoch].append(LB_id)
                M.MiND_id[epoch].append(MiND_id)
                M.MADA_id[epoch].append(MADA_id)
                M.PCA_id[epoch].append(PCA_id)
                find = True
                    
        if not find:
            M = MSE(tau)
            M.mse_act[epoch].append(mse_act)
            M.mse_in[epoch].append(mse_in)
            M.LB_id[epoch].append(LB_id)
            M.MiND_id[epoch].append(MiND_id)
            M.MADA_id[epoch].append(MADA_id)
            M.PCA_id[epoch].append(PCA_id)
            items.append(M)
    fp.close()

    for M in items:
        mse_act_list = []
        mse_in_list = []
        LB_id_list = []
        MiND_id_list = []
        MADA_id_list = []
        PCA_id_list = []
        for epoch in range(max_epoch):
            mse_act_list.append(np.mean(M.mse_act[epoch]))
            mse_in_list.append(np.mean(M.mse_in[epoch]))
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
        plt.plot(range(max_epoch), mse_act_list, label='act')
        plt.plot(range(max_epoch), mse_in_list, label='in')
        plt.legend()
        plt.savefig(f'logs/time-lagged/tau_{tau}/ID_per_epoch.jpg', dpi=300)
        plt.close()
        
        
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


def plot_fhn_autocorr():

    data = np.load('../Data/origin/lattice_boltzmann.npz')
    rho_act_all = np.array(data["rho_act_all"])[0]
    rho_in_all = np.array(data["rho_in_all"])[0]

    data_1 = pd.DataFrame(rho_act_all, columns=[f'x_{i}' for i in range(rho_act_all.shape[-1])])
    data_2 = pd.DataFrame(rho_in_all, columns=[f'x_{i}' for i in range(rho_act_all.shape[-1])])
    
    plt.figure(figsize=(12,6))
    for item_i, pack in enumerate(zip(['act', 'in'], [data_1, data_2])):
        
        item, data = pack[0], pack[1]
        
        X, T, Corr = [], [], []
        lag_list = np.arange(0, rho_act_all.shape[0], 1000)
        for lag in tqdm(lag_list):
            # for i, x in enumerate(np.linspace(0, 20, rho_act_all.shape[-1])):
            for i, x in enumerate(np.linspace(0, 20, 20)):
                X.append(x)
                T.append(lag*0.001)
                Corr.append(data[f'x_{i}'].autocorr(lag=lag))
        
        loc = [0.05+0.5*item_i, 0.05, 0.4, 0.9] # left, bottom, width, height
        ax = plt.axes(loc, projection='3d')
        pic = ax.scatter(X, T, Corr, s=5, c=Corr, cmap=plt.get_cmap("coolwarm")) # s=size, cmap=color map
        ax.set_xlabel('x', labelpad=15)
        ax.set_ylabel(f't', labelpad=15)
        ax.set_zlabel('corr', labelpad=15)
        ax.set_title(item)
        
        np.savez(item+'_corr.npz', X=X, T=T, Corr=Corr)
        
    plt.savefig('corr.jpg', dpi=300)
    plt.close()

    
def plot_contourf_fhn(data, tau, path):
    """画一张热力图"""
    
    # data: (time_length, 2, 101)
    time_length = data.shape[0]
    x_length = data.shape[2]
    
    fig = plt.figure(figsize=(12,6))
    for index, item in enumerate(['act', 'in']):
        
        X, Y = np.meshgrid(np.arange(x_length), np.arange(0, time_length)*tau)
        Z = data[:, index]
        
        loc = [0.05+0.5*index, 0.05, 0.4, 0.95] # left, bottom, width, height
        ax = plt.axes(loc)
        pic = ax.contourf(X, Y, Z, 100, cmap=plt.get_cmap("coolwarm"), zorder=-9)
        ax.set_ylabel(r"$t$")
        ax.set_xlabel(r"$x$")
        ax.set_title(item)
        fig.colorbar(pic)
        plt.gca().set_rasterization_zorder(-1)
    
    plt.savefig(path+f".jpg", bbox_inches="tight", dpi=300)
    plt.close()
   
    
def plot_contourf_fhn_pred(pred, true, diff, tau, path):
    """一次把pred、true、diff的热力图画在一张图里"""
    
    fig = plt.figure(figsize=(16,15))
    for index, data in enumerate([true, pred, diff]):        
        for item_i, item in enumerate(['act', 'in']):
            
            time_length = data.shape[0]
            x_length = data.shape[2]
            X, Y = np.meshgrid(np.arange(x_length), np.arange(0, time_length)*tau)
            Z = data[:, item_i]
            
            loc = [0.05+0.5*item_i, 1.0-0.3*(index+1), 0.4, 0.2] # left, bottom, width, height
            ax = plt.axes(loc)
            pic = ax.contourf(X, Y, Z, 100, cmap=plt.get_cmap("coolwarm"), zorder=-9)
            ax.set_ylabel(r"$t$")
            ax.set_xlabel(r"$x$")
            ax.set_title(['true','pred','diff'][index]+' | '+item)
            fig.colorbar(pic)
            plt.gca().set_rasterization_zorder(-1)
    
    fig.savefig(path+".jpg", dpi=300)
    plt.close()
    
    
def plot_contourf_fhn_slow_fast(slow, fast, tau, path):
    """一次把快、慢数据的热力图画在一张图里"""
    
    fig = plt.figure(figsize=(16,10))
    for index, data in enumerate([slow, fast]):        
        for item_i, item in enumerate(['act', 'in']):
            
            time_length = data.shape[0]
            x_length = data.shape[2]
            X, Y = np.meshgrid(np.arange(x_length), np.arange(0, time_length)*tau)
            Z = data[:, item_i]
            
            loc = [0.05+0.5*item_i, 1.0-0.45*(index+1), 0.4, 0.35] # left, bottom, width, height
            ax = plt.axes(loc)
            pic = ax.contourf(X, Y, Z, 100, cmap=plt.get_cmap("coolwarm"), zorder=-9)
            ax.set_ylabel(r"$t$")
            ax.set_xlabel(r"$x$")
            ax.set_title(['slow','fast'][index]+' | '+item)
            fig.colorbar(pic)
            plt.gca().set_rasterization_zorder(-1)
    
    fig.savefig(path+".jpg", dpi=300)
    plt.close()


if __name__ == '__main__':
    
    plot_fhn_autocorr()