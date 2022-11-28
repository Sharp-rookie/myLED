import pickle
import numpy as np
import os
import h5py

# Load and concat data
tau = 0.005
with open(f"./Simulation_Data/lattice_boltzmann_fhn_tau{tau}.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    simdata = pickle.load(file)
    rho_act_all = np.array(simdata["rho_act_all"])
    rho_in_all = np.array(simdata["rho_in_all"])
    dt_coarse = simdata["dt_data"]
    del simdata
rho_act_all = np.expand_dims(rho_act_all, axis=2)
rho_in_all = np.expand_dims(rho_in_all, axis=2)
sequences_raw = np.concatenate((rho_act_all, rho_in_all), axis=2)
print('activator shape', np.shape(rho_act_all))
print('inhibitor shape', np.shape(rho_in_all))
print('concat shape', np.shape(sequences_raw))

# Save statistic information
data_dir_scaler = f"./Data_tau{tau}"
os.makedirs(data_dir_scaler, exist_ok=True)
data_max = np.max(sequences_raw, axis=(0,1,3))
data_min = np.min(sequences_raw, axis=(0,1,3))
np.savetxt(data_dir_scaler + "/data_max.txt", data_max)
np.savetxt(data_dir_scaler + "/data_min.txt", data_min)
np.savetxt(data_dir_scaler + "/dt.txt", [dt_coarse]) # Save the timestep


#######################
# Create train data
#######################

# total time steps for train
total_length = 3000
# single-sample time steps for train
sequence_length = 4

# select 3 initial-condition(ICs) traces for train
ICS_TRAIN = [0, 1, 2]
N_ICS_TRAIN=len(ICS_TRAIN)
sequences_raw_train = sequences_raw[ICS_TRAIN, :total_length]

batch_size  = 256

# random select n*batch_size sequence_length-lengthed trace index from 0,1,2 ICs time-series data
idxs_timestep = []
idxs_ic = []
for ic in range(N_ICS_TRAIN):
    seq_data = sequences_raw_train[ic]
    idxs = np.arange(0, np.shape(seq_data)[0]- sequence_length, 1)
    # idxs = np.random.permutation(idxs)
    for idx_ in idxs:
        idxs_ic.append(ic)
        idxs_timestep.append(idx_)
max_batches = int(np.floor(len(idxs_ic)/batch_size))
num_sequences = max_batches*batch_size
print("Number of sequences = {:}/{:}".format(num_sequences, len(idxs_ic)))

# generator train dataset by random index from 3 ICs
sequences = []
for bn in range(num_sequences):
    idx_ic = idxs_ic[bn]
    idx_timestep = idxs_timestep[bn]
    sequence = sequences_raw_train[idx_ic, idx_timestep:idx_timestep+sequence_length]
    sequences.append(sequence)

sequences = np.array(sequences) 
print("Train Dataset", np.shape(sequences))

# save train dataset
data_dir = f"./Data_tau{tau}/train"
os.makedirs(data_dir, exist_ok=True)
hf = h5py.File(data_dir + '/data.h5', 'w')
# Only a single sequence_example per dataset group
for seq_num_ in range(np.shape(sequences)[0]):
    data_group = sequences[seq_num_]
    data_group = np.array(data_group)
    gg = hf.create_group('batch_{:010d}'.format(seq_num_))
    gg.create_dataset('data', data=data_group)
hf.close()


#######################
# Create valid data
#######################

# total time steps for val
total_length = 3000
# single-sample time steps for val
sequence_length = 4

# select 2 initial-condition traces for val
ICS_VAL = [3, 4]
N_ICS_VAL = len(ICS_VAL)
sequences_raw_val = sequences_raw[ICS_VAL, :total_length]

batch_size  = 256

# random select n*batch_size sequence_length-lengthed trace index from 3,4 ICs time-series data
idxs_timestep = []
idxs_ic = []
for ic in range(N_ICS_VAL):
    seq_data = sequences_raw_val[ic]
    idxs = np.arange(0, np.shape(seq_data)[0]- sequence_length, 1)
    # idxs = np.random.permutation(idxs)

    for idx_ in idxs:
        idxs_ic.append(ic)
        idxs_timestep.append(idx_)
max_batches = int(np.floor(len(idxs_ic)/batch_size))
print("Number of sequences = {:}/{:}".format(max_batches*batch_size, len(idxs_ic)))
num_sequences = max_batches*batch_size

# generator val dataset by random index from 2 ICs
sequences = []
for bn in range(num_sequences):
    idx_ic = idxs_ic[bn]
    idx_timestep = idxs_timestep[bn]
    sequence = sequences_raw_val[idx_ic, idx_timestep:idx_timestep+sequence_length]
    sequences.append(sequence)
sequences = np.array(sequences) 
print("Val Dataset", np.shape(sequences))

# save val dataset
data_dir = f"./Data_tau{tau}/val"
os.makedirs(data_dir, exist_ok=True)
hf = h5py.File(data_dir + '/data.h5', 'w')
# Only a single sequence_example per dataset group
for seq_num_ in range(np.shape(sequences)[0]): # (960, 121, 202)
    data_group = sequences[seq_num_]
    data_group = np.array(data_group)
    gg = hf.create_group('batch_{:010d}'.format(seq_num_))
    gg.create_dataset('data', data=data_group)
hf.close()


#######################
# Create test data
#######################

# total time steps for test
total_length = 3000
# single-sample time steps for test
sequence_length = 4

# select 2 initial-condition traces for test
ICS_TEST = [5]
N_ICS_TEST = len(ICS_TEST)
sequences_raw_test = sequences_raw[ICS_TEST, :total_length][0]

batch_size = 1

# random select n*batch_size sequence_length-lengthed trace index from 5, ICs time-series data
idxs = np.arange(0, np.shape(sequences_raw_test)[0]- sequence_length, 1)
# idxs = np.random.permutation(idxs)
num_sequences = len(idxs)

# generator test dataset by random index from 2 ICs
sequences = []
for i in range(num_sequences):
    idx_timestep = idxs[i]
    sequence = sequences_raw_test[idx_timestep:idx_timestep+sequence_length]
    sequences.append(sequence)
sequences = np.array(sequences) 
print("Test Dataset", np.shape(sequences))

# save test dataset
data_dir = f"./Data_tau{tau}/test"
os.makedirs(data_dir, exist_ok=True)
hf = h5py.File(data_dir + '/data.h5', 'w')
# Only a single sequence_example per dataset group
for seq_num_ in range(np.shape(sequences)[0]):
    data_group = sequences[seq_num_]
    data_group = np.array(data_group)
    gg = hf.create_group('batch_{:010d}'.format(seq_num_))
    gg.create_dataset('data', data=data_group)
hf.close()