import h5py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class scaler(object):
    def __init__(
        self,
        scaler_type,
        data_min,
        data_max,
        common_scaling_per_input_dim=False,
        common_scaling_per_channels=False,
        channels=0,
    ):
        super(scaler, self).__init__()
        # Data are of three possible types:
        # Type 1:
        #         (T, input_dim)
        # Type 2:
        #         (T, input_dim, Cx) (input_dim is the N_particles, perm_inv, etc.)
        # Type 3:
        #         (T, input_dim, Cx, Cy, Cz)
        self.scaler_type = scaler_type

        if self.scaler_type not in ["MinMaxZeroOne", "MinMaxMinusOneOne"]:
            raise ValueError("Scaler {:} not implemented.".format(
                self.scaler_type))

        self.data_min = np.array(data_min)
        self.data_max = np.array(data_max)
        self.data_range = self.data_max - self.data_min
        self.common_scaling_per_input_dim = common_scaling_per_input_dim
        self.common_scaling_per_channels = common_scaling_per_channels
        self.channels = channels

    def scaleData(self,
                  batch_of_sequences,
                  reuse=None,
                  single_sequence=False,
                  check_bounds=True):
        if single_sequence: batch_of_sequences = batch_of_sequences[np.newaxis]
        # Size of the batch_of_sequences is [K, T, ...]
        # Size of the batch_of_sequences is [K, T, D]
        # Size of the batch_of_sequences is [K, T, D, C]
        # Size of the batch_of_sequences is [K, T, D, C, C]
        # Size of the batch_of_sequences is [K, T, D, C, C, C]
        self.data_shape = np.shape(batch_of_sequences)
        self.data_shape_length = len(self.data_shape)

        if self.scaler_type == "MinMaxZeroOne":
            data_min = self.repeatScalerParam(self.data_min, self.data_shape)
            data_max = self.repeatScalerParam(self.data_max, self.data_shape)

            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_min)))
            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_max)))
            batch_of_sequences_scaled = np.array(
                (batch_of_sequences - data_min) / (data_max - data_min))

            if check_bounds:
                assert (np.all(batch_of_sequences_scaled >= 0.0))
                assert (np.all(batch_of_sequences_scaled <= 1.0))

        elif self.scaler_type == "MinMaxMinusOneOne":
            data_min = self.repeatScalerParam(self.data_min, self.data_shape)
            data_max = self.repeatScalerParam(self.data_max, self.data_shape)

            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_min)))
            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_max)))

            batch_of_sequences_scaled = np.array(
                (2.0 * batch_of_sequences - data_max - data_min) /
                (data_max - data_min))

            if check_bounds:
                assert (np.all(batch_of_sequences_scaled >= -1.0))
                assert (np.all(batch_of_sequences_scaled <= 1.0))

        elif self.scaler_type == "Standard":
            data_mean = self.repeatScalerParam(
                self.data_mean, self.data_shape)
            data_std = self.repeatScalerParam(
                self.data_std, self.data_shape)

            assert (np.all(
                np.shape(batch_of_sequences) == np.shape(data_mean)))
            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_std)))

            batch_of_sequences_scaled = np.array(
                (batch_of_sequences - data_mean) / data_std)

        else:
            raise ValueError("Scaler not implemented.")

        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[0]
        return batch_of_sequences_scaled

    def repeatScalerParam(self, data, shape):
        # Size of the batch_of_sequences is [K, T, ...]
        # Size of the batch_of_sequences is [K, T, D]
        # Size of the batch_of_sequences is [K, T, D, C]
        # Size of the batch_of_sequences is [K, T, D, C, C]
        # Size of the batch_of_sequences is [K, T, D, C, C, C]

        common_scaling_per_channels = self.common_scaling_per_channels
        common_scaling_per_input_dim = self.common_scaling_per_input_dim
        # Running through the shape in reverse order
        if common_scaling_per_input_dim:
            D = shape[2]
            # Commong scaling for all inputs !
            data = np.repeat(data[np.newaxis], D, 0)

        # Running through the shape in reverse order
        if common_scaling_per_channels:
            # Repeating the scaling for each channel
            assert (len(shape[::-1][:-3]) == self.channels)
            for channel_dim in shape[::-1][:-3]:
                data = np.repeat(data[np.newaxis], channel_dim, 0)
                data = np.swapaxes(data, 0, 1)

        T = shape[1]
        data = np.repeat(data[np.newaxis], T, 0)
        K = shape[0]
        data = np.repeat(data[np.newaxis], K, 0)
        return data

    def descaleData(self,
                    batch_of_sequences_scaled,
                    single_sequence=True,
                    single_batch=False,
                    verbose=True,
                    check_bounds=True):

        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[np.newaxis]
        if single_batch:
            batch_of_sequences_scaled = batch_of_sequences_scaled[np.newaxis]

        # Size of the batch_of_sequences_scaled is [K, T, ...]
        # Size of the batch_of_sequences_scaled is [K, T, D]
        # Size of the batch_of_sequences_scaled is [K, T, D, C]
        # Size of the batch_of_sequences_scaled is [K, T, D, C, C]
        # Size of the batch_of_sequences_scaled is [K, T, D, C, C, C]
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

            data_mean = self.repeatScalerParam(self.data_mean, self.data_shape)
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


class FHNDataset(Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (train or val).
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
    """

    def __init__(
        self,
        file_path,
        data_cache_size=3,
        data_info_dict=None
    ):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size

        if data_info_dict is not None:
            # Scaler
            self.scaler = data_info_dict["scaler"] if "scaler" in data_info_dict.keys() else None
            # Truncating the timesteps of the dataset
            self.truncate_timesteps = data_info_dict["truncate_timesteps"] \
                if "truncate_timesteps" in data_info_dict else None
            # Truncating the number of batches in the data
            self.truncate_data_batches = data_info_dict["truncate_data_batches"] \
                if "truncate_data_batches" in data_info_dict else None
        else:
            self.truncate_data_batches = None
            self.truncate_timesteps = None
            self.scaler = None

        # Search for all h5 files
        p = Path(file_path)
        assert p.is_dir(), "Path to data files {:} is not found.".format(p)
        files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()))

        if not ((self.truncate_timesteps is None) or (self.truncate_timesteps == 0)):
            print("Loader restricted to {:} timesteps.".format(self.truncate_timesteps))

    # 0 --> 1
    def __getitem__(self, index):

        trace = self.get_data("data", index)

        if self.scaler is not None:
            trace = self.scaler.scaleData(trace, single_sequence=True)
        if not ((self.truncate_timesteps is None) or (self.truncate_timesteps == 0)):
            trace = trace[:self.truncate_timesteps]
        
        trace = torch.from_numpy(trace).float()
        data = trace[0]
        if trace.shape[0] == 2:
            target = trace[1]
        elif trace.shape[0] == 1:
            target = trace[0]

        return data, target

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path):
        
        number_of_loaded_groups = 0
        number_of_total_groups = 0
        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                if self.truncate_data_batches is None or self.truncate_data_batches == 0 or number_of_loaded_groups < self.truncate_data_batches:
                    for dname, ds in group.items():
                        # if data is not loaded its cache index is -1
                        idx = -1
                        # type is derived from the name of the dataset; we expect the dataset
                        # name to have a name such as 'data' or 'label' to identify its type
                        # we also store the shape of the data in case we need it
                        self.data_info.append({
                            'file_path': file_path,
                            'type': dname,
                            'shape': ds[()].shape,
                            'cache_idx': idx
                        })
                    number_of_loaded_groups += 1
                number_of_total_groups += 1

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """

        number_of_loaded_groups = 0
        number_of_total_groups = 0
        with h5py.File(file_path, 'r') as h5_file:
            for gname, group in h5_file.items():
                if self.truncate_data_batches is None or self.truncate_data_batches == 0 or number_of_loaded_groups < self.truncate_data_batches:

                    for dname, ds in group.items():
                        # add data to the data cache and retrieve
                        # the cache index
                        idx = self._add_to_cache(ds[()], file_path)

                        # find the beginning index of the hdf5 file we are looking for
                        file_idx = next(i for i, v in enumerate(self.data_info)
                                        if v['file_path'] == file_path)

                        # the data info should have the same index since we loaded it in the same way
                        self.data_info[file_idx + idx]['cache_idx'] = idx

                    number_of_loaded_groups += 1

                number_of_total_groups += 1

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{
                'file_path': di['file_path'],
                'type': di['type'],
                'shape': di['shape'],
                'cache_idx': -1
            } if di['file_path'] == removal_keys[0] else di
                              for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """

        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """

        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """

        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


if __name__=='__main__':

    data_info_dict = {
        # 'truncate_data_batches': 2048, 
        'scaler': scaler(
                scaler_type='MinMaxZeroOne',
                data_min=np.loadtxt("Data/Data/tau_0.5/data_min.txt"),
                data_max=np.loadtxt("Data/Data/tau_0.5/data_max.txt"),
                channels=1,
                common_scaling_per_input_dim=0,
                common_scaling_per_channels=1,  # Common scaling for all channels
            )
        }
    dataset = FHNDataset(
            'Data/Data/tau_0.5/val',
            data_cache_size=3,
            data_info_dict=data_info_dict
        )

    data, target = dataset.__getitem__(0)
    print(len(dataset), data.shape, target.shape)
   
    plot = True
    if plot:
        import matplotlib.pyplot as plt
        dimension=85 # (0, 101)
        data_act = []
        data_in = []
        target_act = []
        target_in = []
        for i in range(len(dataset)):
            data, target = dataset.__getitem__(i)
            data_act.append(data[0, dimension])
            data_in.append(data[1, dimension])
            target_act.append(target[0, dimension])
            target_in.append(target[1, dimension])
        plt.figure()
        plt.scatter(np.array([i*0.5 for i in range(len(dataset))]), np.array(data_act), label='input activator')
        plt.scatter(np.array([i*0.5 for i in range(len(dataset))]), np.array(data_in), label='input inhibitor')
        plt.scatter(np.array([i*0.5 for i in range(len(dataset))]), np.array(target_act), label='target activator')
        plt.scatter(np.array([i*0.5 for i in range(len(dataset))]), np.array(target_in), label='target inhibitor')
        plt.legend()
        plt.savefig(f'dimension{dimension}.jpg', dpi=300)