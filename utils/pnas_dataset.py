import os
import torch
import numpy as np
from torch.utils.data import Dataset


class scaler(object):
    def __init__(
        self,
        scaler_type,
        data_min=None,
        data_max=None,
        data_std=None,
        data_mean=None,
    ):
        super(scaler, self).__init__()
        self.scaler_type = scaler_type

        if self.scaler_type not in ["MinMaxZeroOne", "MinMaxMinusOneOne", "Standard"]:
            raise ValueError("Scaler {:} not implemented.".format(self.scaler_type))

        self.data_min = np.array(data_min)[np.newaxis] if data_min is not None else None
        self.data_max = np.array(data_max)[np.newaxis] if data_max is not None else None
        self.data_mean = np.array(data_mean)[np.newaxis] if data_mean is not None else None
        self.data_std = np.array(data_std)[np.newaxis] if data_std is not None else None

    def scaleData(self,
                  batch_of_sequences,
                  single_sequence=False,
                  check_bounds=True):
        if single_sequence: batch_of_sequences = batch_of_sequences[np.newaxis]

        if self.scaler_type == "MinMaxZeroOne":

            assert (self.data_max is not None and self.data_min is not None)
            assert self.data_min.shape == batch_of_sequences.shape
            assert self.data_max.shape == batch_of_sequences.shape
            
            batch_of_sequences_scaled = np.array((batch_of_sequences - self.data_min) / (self.data_max - self.data_min))

            if check_bounds:
                assert (np.all(batch_of_sequences_scaled >= 0.0)), f'{batch_of_sequences_scaled[batch_of_sequences_scaled < 0.0]}'
                assert (np.all(batch_of_sequences_scaled <= 1.0)), f'{batch_of_sequences_scaled[batch_of_sequences_scaled > 1.0]}'

        elif self.scaler_type == "MinMaxMinusOneOne":

            assert (self.data_max is not None and self.data_min is not None)
            assert self.data_min.shape == batch_of_sequences.shape
            assert self.data_max.shape == batch_of_sequences.shape
            
            batch_of_sequences_scaled = np.array((2.0 * batch_of_sequences - self.data_max - self.data_min) / (self.data_max - self.data_min))

            if check_bounds:
                assert (np.all(batch_of_sequences_scaled >= -1.0)), f'{batch_of_sequences_scaled[batch_of_sequences_scaled < -1.0]}'
                assert (np.all(batch_of_sequences_scaled <= 1.0)), f'{batch_of_sequences_scaled[batch_of_sequences_scaled > 1.0]}'

        elif self.scaler_type == "Standard":
            
            assert (self.data_mean is not None and self.data_std is not None)
            assert self.data_mean.shape == batch_of_sequences.shape
            assert self.data_std.shape == batch_of_sequences.shape

            batch_of_sequences_scaled = np.array((batch_of_sequences - self.data_mean) / self.data_std)

        else:
            raise ValueError("Scaler not implemented.")

        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[0]
        return batch_of_sequences_scaled

    def descaleData(self,
                    batch_of_sequences_scaled,
                    single_sequence=True,
                    single_batch=False,
                    check_bounds=True):

        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[np.newaxis]
        if single_batch:
            batch_of_sequences_scaled = batch_of_sequences_scaled[np.newaxis]
        self.data_shape = np.shape(batch_of_sequences_scaled)
        self.data_shape_length = len(self.data_shape)

        if self.scaler_type == "MinMaxZeroOne":

            data_min = self.repeatScalerParam(self.data_min, self.data_shape)
            data_max = self.repeatScalerParam(self.data_max, self.data_shape)

            assert (np.all(np.shape(batch_of_sequences_scaled) == np.shape(data_min)))
            assert (np.all(np.shape(batch_of_sequences_scaled) == np.shape(data_max)))

            if isinstance(batch_of_sequences_scaled, torch.Tensor):
                batch_of_sequences = np.array(batch_of_sequences_scaled.cpu() * (data_max - data_min) + data_min)
            else:
                batch_of_sequences = np.array(batch_of_sequences_scaled * (data_max - data_min) + data_min)

            if check_bounds:
                assert (np.all(batch_of_sequences >= data_min))
                assert (np.all(batch_of_sequences <= data_max))

        elif self.scaler_type == "MinMaxMinusOneOne":

            data_min = self.repeatScalerParam(self.data_min, self.data_shape)
            data_max = self.repeatScalerParam(self.data_max, self.data_shape)

            assert (np.all(np.shape(batch_of_sequences_scaled) == np.shape(data_min)))
            assert (np.all(np.shape(batch_of_sequences_scaled) == np.shape(data_max)))

            if isinstance(batch_of_sequences_scaled, torch.Tensor):
                batch_of_sequences = np.array(batch_of_sequences_scaled.cpu() * (data_max - data_min) + data_min + data_max) / 2.0
            else:
                batch_of_sequences = np.array(batch_of_sequences_scaled * (data_max - data_min) + data_min + data_max) / 2.0

            if check_bounds:
                assert (np.all(batch_of_sequences >= data_min))
                assert (np.all(batch_of_sequences <= data_max))

        elif self.scaler_type == "Standard":

            data_mean = self.repeatScalerParam(data_mean, self.data_shape)
            data_std = self.repeatScalerParam(self.data_std, self.data_shape)

            assert (np.all(np.shape(batch_of_sequences_scaled) == np.shape(data_mean)))
            assert (np.all(np.shape(batch_of_sequences_scaled) == np.shape(data_std)))

            if isinstance(batch_of_sequences_scaled, torch.Tensor):
                batch_of_sequences = np.array(batch_of_sequences_scaled.cpu() * data_std + data_mean)
            else:
                batch_of_sequences = np.array(batch_of_sequences_scaled * data_std + data_mean)

        else:
            raise ValueError("Scaler not implemented.")

        if single_sequence: batch_of_sequences = batch_of_sequences[0]
        if single_batch: batch_of_sequences = batch_of_sequences[0]
        return np.array(batch_of_sequences)


class PNASDataset(Dataset):

    def __init__(
        self,
        file_path,
        input_scaler=None,
        target_scaler=None,
    ):
        super().__init__()

        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

        # Search for txt files
        self.data = np.load(file_path)['data'] # (N, 2, 3)

    # 0 --> 1
    def __getitem__(self, index):

        trace = self.data[index]

        input = trace[0]
        target = trace[0] if len(trace)==1 else trace[1]
        # target = trace[1] - trace[0] # diff

        if self.input_scaler is not None:
            input = self.input_scaler.scaleData(input, single_sequence=True)
        if self.target_scaler is not None:
            target = self.target_scaler.scaleData(target, single_sequence=True)

        input = torch.from_numpy(input[np.newaxis]).float()
        target = torch.from_numpy(target[np.newaxis]).float()

        return input, target

    def __len__(self):
        return len(self.data)


if __name__=='__main__':

    tau = 0.005
    input_scaler = scaler(
                        scaler_type='MinMaxZeroOne',
                        data_min=np.loadtxt(f"Data/data/tau_{tau}/data_min.txt"),
                        data_max=np.loadtxt(f"Data/data/tau_{tau}/data_max.txt"),
            )
    target_scaler = scaler(
                        scaler_type='MinMaxZeroOne',
                        data_min=np.loadtxt(f"Data/data/tau_{tau}/data_min.txt"),
                        data_max=np.loadtxt(f"Data/data/tau_{tau}/data_max.txt"),
            )
    dataset = PNASDataset(
        f'Data/data/tau_{tau}/val.npz', 
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        )

    data, target = dataset.__getitem__(0)
    print(len(dataset), data.shape, target.shape)
   
    plot = True
    if plot:
        import matplotlib.pyplot as plt
        X = []
        Y = []
        Z = []
        for i in range(len(dataset)):
            data, target = dataset.__getitem__(i)
            X.append([data[0][0], target[0][0]])
            Y.append([data[0][1], target[0][1]])
            Z.append([data[0][2], target[0][2]])
        plt.figure(figsize=(16,9))
        ax1 = plt.subplot(2,3,1)
        ax1.set_title('input X')
        plt.plot(np.array([i*tau for i in range(len(dataset))]), np.array(X)[:,0])
        plt.xlabel('time / s')
        ax2 = plt.subplot(2,3,2)
        ax2.set_title('input Y')
        plt.plot(np.array([i*tau for i in range(len(dataset))]), np.array(Y)[:,0])
        plt.xlabel('time / s')
        ax3 = plt.subplot(2,3,3)
        ax3.set_title('input Z')
        plt.plot(np.array([i*tau for i in range(len(dataset))]), np.array(Z)[:,0])
        plt.xlabel('time / s')
        ax4 = plt.subplot(2,3,4)
        ax4.set_title('target X')
        plt.plot(np.array([i*tau for i in range(len(dataset))]), np.array(X)[:,1])
        plt.xlabel('time / s')
        ax5 = plt.subplot(2,3,5)
        ax5.set_title('target Y')
        plt.plot(np.array([i*tau for i in range(len(dataset))]), np.array(Y)[:,1])
        plt.xlabel('time / s')
        ax6 = plt.subplot(2,3,6)
        ax6.set_title('target Z')
        plt.plot(np.array([i*tau for i in range(len(dataset))]), np.array(Z)[:,1])
        plt.xlabel('time / s')

        plt.subplots_adjust(left=0.1,
            right=0.9,
            top=0.9,
            bottom=0.15,
            wspace=0.2,
            hspace=0.35,
        )
        plt.savefig(f'dataset.jpg', dpi=300)