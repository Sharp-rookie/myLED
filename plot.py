import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_id():
    fp = open('ID.txt', 'r')
    tau_list = []
    id_list = []
    for id_str in fp.readlines():
        tau = id_str[:-1].split('--')[0]
        id = id_str[:-1].split('--')[1]
        tau_list.append(float(tau))
        id_list.append(float(id))
    index = np.argsort(tau_list)
    tau_list = np.array(tau_list)[index]
    id_list = np.array(id_list)[index]
    plt.figure()
    plt.xlabel('tau/s')
    plt.ylabel('ID')
    plt.plot(tau_list, id_list)
    plt.scatter(tau_list, id_list)
    plt.savefig(f'ID.jpg', dpi=300)


def plot_val_mse():

    class MSE():
        def __init__(self, tau):
            self.tau = tau
            self.mse_x = []
            self.mse_y = []
            self.mse_z = []

    fp = open('val_mse.txt', 'r')
    items = []
    for line in fp.readlines():
        tau = float(line[:-1].split(',')[0])
        seed = int(line[:-1].split(',')[1])
        mse_x = float(line[:-1].split(',')[2])
        mse_y = float(line[:-1].split(',')[3])
        mse_z = float(line[:-1].split(',')[4])

        find = False
        for M in items:
            if M.tau == tau:
                M.mse_x.append(mse_x)
                M.mse_y.append(mse_y)
                M.mse_z.append(mse_z)
                find = True
                    
        if not find:
            M = MSE(tau)
            M.mse_x.append(mse_x)
            M.mse_y.append(mse_y)
            M.mse_z.append(mse_z)
            items.append(M)

    tau_list = []
    mse_x_list = []
    mse_y_list = []
    mse_z_list = []
    for M in items:
        tau_list.append(M.tau)
        mse_x_list.append(np.mean(M.mse_x))
        mse_y_list.append(np.mean(M.mse_y))
        mse_z_list.append(np.mean(M.mse_z))

    plt.figure()
    plt.xlabel('tau/s')
    plt.ylabel('MSE')
    plt.scatter(tau_list, mse_x_list, label='x')
    plt.scatter(tau_list, mse_y_list, label='y')
    plt.scatter(tau_list, mse_z_list, label='z')
    plt.ylim((0., 1.05*max(np.max(mse_x_list), np.max(mse_y_list), np.max(mse_z_list))))
    plt.legend()
    plt.savefig(f'val_mse.jpg', dpi=300)


def plot_epoch_test_log():

    os.makedirs('plot/', exist_ok=True)

    max_epoch = 500+1
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

    fp = open('test_log.txt', 'r')
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
        plt.savefig(f'plot/test_tau{M.tau:.3f}.jpg', dpi=300)
        plt.close()


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

def plot_dataset(tau):

    if os.path.exists(f"Data/Simulation_Data/lattice_boltzmann_fhn_original.pickle"):
        print('wait for loading data ...')
        with open(f"Data/Simulation_Data/lattice_boltzmann_fhn_original.pickle", "rb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            simdata = pickle.load(file)
            rho_act_all = np.array(simdata["rho_act_all"])
            rho_in_all = np.array(simdata["rho_in_all"])
            del simdata
    else:
        exit(0)

    # -------------------------------- 2_create_training_data -------------------------------- 

    # subsampling
    dt = 0.005
    subsampling = int(tau/dt)
    rho_act_all_n = []
    rho_in_all_n = []
    for rho_act, rho_in in zip(rho_act_all, rho_in_all):
        rho_act = rho_act[::subsampling]
        rho_in = rho_in[::subsampling]
        rho_act_all_n.append(rho_act)
        rho_in_all_n.append(rho_in)

    rho_act_all = np.expand_dims(rho_act_all_n, axis=2)
    rho_in_all = np.expand_dims(rho_in_all_n, axis=2)
    sequences_raw = np.concatenate((rho_act_all, rho_in_all), axis=2)
    print('activator shape', np.shape(rho_act_all))
    print('inhibitor shape', np.shape(rho_in_all))
    print('concat shape', np.shape(sequences_raw))

    # single-sample time steps for train
    sequence_length = 1

    batch_size = 1

    #######################
    # Create valid data
    #######################

    # select 1 initial-condition traces for val
    ICS_VAL = [3]
    N_ICS_VAL = len(ICS_VAL)
    sequences_raw_val = sequences_raw[ICS_VAL]

    # random select n*batch_size sequence_length-lengthed trace index from 3,4 ICs time-series data
    idxs_timestep = []
    idxs_ic = []
    for ic in range(N_ICS_VAL):
        seq_data = sequences_raw_val[ic]
        idxs = np.arange(0, np.shape(seq_data)[0]- sequence_length, 1)

        for idx_ in idxs:
            idxs_ic.append(ic)
            idxs_timestep.append(idx_)
    max_batches = int(np.floor(len(idxs_ic)/batch_size))
    print("Number of sequences = {:}/{:}".format(max_batches*batch_size, len(idxs_ic)))
    num_sequences = max_batches*batch_size

    # generator val dataset by random index from 1 ICs
    sequences = []
    for bn in range(num_sequences):
        idx_ic = idxs_ic[bn]
        idx_timestep = idxs_timestep[bn]
        sequence = sequences_raw_val[idx_ic, idx_timestep:idx_timestep+sequence_length]
        sequences.append(sequence)
    sequences = np.array(sequences) 
    print("Val Dataset", np.shape(sequences))

    # continuous
    plt.figure(figsize=(16,10))
    ax1 = plt.subplot(3,1,1)
    ax1.set_title('grid = 25')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:,0,0,25], label='act')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:,0,1,25], label='in')
    plt.legend()
    ax2 = plt.subplot(3,1,2)
    ax2.set_title('grid = 50')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:,0,0,50], label='act')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:,0,1,50], label='in')
    plt.legend()
    ax3 = plt.subplot(3,1,3)
    ax3.set_title('grid = 85')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:,0,0,85], label='act')
    plt.plot([i*tau for i in range(len(sequences))], sequences[:,0,1,85], label='in')
    plt.legend()
    plt.title(f'tau = {tau}')
    plt.savefig(f'Analyse/data_tau{tau}.jpg', dpi=500)

    # discrete
    part = (np.array([0, 200])/tau).astype(np.int32)
    plt.figure(figsize=(16,10))
    ax1 = plt.subplot(3,1,1)
    ax1.set_title('grid = 25')
    plt.scatter([i*tau for i in range(*part)], sequences[part[0]:part[1],0,0,25], label='act')
    plt.scatter([i*tau for i in range(*part)], sequences[part[0]:part[1],0,1,25], label='in')
    plt.legend()
    ax2 = plt.subplot(3,1,2)
    ax2.set_title('grid = 50')
    plt.scatter([i*tau for i in range(*part)], sequences[part[0]:part[1],0,0,50], label='act')
    plt.scatter([i*tau for i in range(*part)], sequences[part[0]:part[1],0,1,50], label='in')
    plt.legend()
    ax3 = plt.subplot(3,1,3)
    ax3.set_title('grid = 85')
    plt.scatter([i*tau for i in range(*part)], sequences[part[0]:part[1],0,0,85], label='act')
    plt.scatter([i*tau for i in range(*part)], sequences[part[0]:part[1],0,1,85], label='in')
    plt.legend()
    plt.title(f'tau = {tau}')
    plt.savefig(f'Analyse/data_tau{tau}_part.jpg', dpi=500)


if __name__ == '__main__':

    # os.makedirs('Analyse/', exist_ok=True)
    # [plot_dataset(tau=tau) for tau in np.arange(0.1, 0.75, 0.05)]

    # plot_id()
    # plot_val_mse()
    
    plot_epoch_test_log()
    # plot_y_corr()