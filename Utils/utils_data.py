import h5py
import torch
import pickle
import numpy as np
from pathlib import Path
from torch.utils import data


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        scaler: Scaler transform to apply to every data instance (default=None).
    """
    def __init__(
        self,
        file_path,
        recursive,
        load_data,
        data_cache_size=3,
        data_info_dict=None,
        rank=0,
    ):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size

        if data_info_dict is not None:

            # Scaler
            if "scaler" in data_info_dict.keys():
                self.scaler = data_info_dict["scaler"]
            else:
                self.scaler = None

            # Truncating the timesteps of the dataset
            if "truncate_timesteps" in data_info_dict:
                self.truncate_timesteps = data_info_dict["truncate_timesteps"]
            else:
                self.truncate_timesteps = None

            # Truncating the number of batches in the data
            if "truncate_data_batches" in data_info_dict:
                self.truncate_data_batches = data_info_dict[
                    "truncate_data_batches"]
            else:
                self.truncate_data_batches = None

            # self.data_info_dict = data_info_dict
        else:
            self.truncate_data_batches = None
            self.truncate_timesteps = None
            self.scaler = None

        # Search for all h5 files
        p = Path(file_path)
        assert p.is_dir(), "Path to data files {:} is not found.".format(p)
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('[utils_data] No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

        if not ((self.truncate_timesteps is None) or(self.truncate_timesteps == 0)):
            print("[utils_data] Loader restricted to {:} timesteps.".format(self.truncate_timesteps))

    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        # print(np.shape(x))
        # print(np.shape(self.scaler.data_min))

        if self.scaler is not None:
            x = self.scaler.scaleData(x, single_sequence=True)

        if not ((self.truncate_timesteps is None) or(self.truncate_timesteps == 0)):
            x = x[:self.truncate_timesteps]

        # print(index)
        # x = index
        return x

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path, load_data):
        number_of_loaded_groups = 0
        number_of_total_groups = 0
        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():

                if self.truncate_data_batches is None or self.truncate_data_batches == 0 or number_of_loaded_groups < self.truncate_data_batches:
                    for dname, ds in group.items():
                        # if data is not loaded its cache index is -1
                        idx = -1
                        if load_data:
                            # add data to the data cache
                            idx = self._add_to_cache(ds[()], file_path)
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


class HDF5DatasetStructured(data.Dataset):
    def __init__(self, file_path, data_info_dict=None,):
        super().__init__()

        self.seq_paths = []

        # Search for all h5 files
        p = Path(file_path)
        assert p.is_dir(), "Path to data files {:} is not found.".format(p)
        files = sorted(p.glob('*.h5'))
        if len(files) != 1:
            raise RuntimeError('[utils_data] No or more than one hdf5 datasets found')
        file_path = files[0]
        self.h5_file = h5py.File(file_path, 'r')

        if data_info_dict is not None:

            # Scaler
            if "scaler" in data_info_dict.keys():
                self.scaler = data_info_dict["scaler"]
            else:
                self.scaler = None

            # Truncating the samples of the daat
            if "truncate_timesteps" in data_info_dict:
                self.truncate_timesteps = data_info_dict["truncate_timesteps"]
            else:
                self.truncate_timesteps = None

            # Truncating the number of batches in the data
            if "truncate_data_batches" in data_info_dict:
                self.truncate_data_batches = data_info_dict[
                    "truncate_data_batches"]
            else:
                self.truncate_data_batches = None

            # self.data_info_dict = data_info_dict
        else:
            self.truncate_data_batches = None
            self.truncate_timesteps = None
            self.scaler = None
        """ Adding the sequence paths """
        self._add_seq_paths()

    def __delete__(self):
        self.h5_file.close()

    def __len__(self):
        return len(self.seq_paths)

    def _add_seq_paths(self):
        idx = 0
        number_of_loaded_groups = 0
        for gname_seq, group_seq in self.h5_file.items():

            if (self.truncate_data_batches is
                    None) or (self.truncate_data_batches
                              == 0) or idx < self.truncate_data_batches:

                # print(gname_seq)
                # print(idx)
                timesteps = []
                idx_time = 0
                number_of_loaded_timesteps = 0
                for gname_time, group_time in group_seq.items():
                    if (self.truncate_timesteps is None) or (
                            self.truncate_timesteps == 0
                    ) or number_of_loaded_timesteps < self.truncate_timesteps:
                        # print(gname_time)
                        timesteps.append(gname_time)
                        number_of_loaded_timesteps += 1
                    idx_time += 1
                assert len(timesteps) == number_of_loaded_timesteps

                # print(timesteps)
                self.seq_paths.append({
                    'gname_seq': gname_seq,
                    'timesteps': timesteps,
                    'idx': idx,
                    'num_timesteps': len(timesteps),
                })
                number_of_loaded_groups += 1
            idx += 1

        print("[utils_data] Loader restricted to {:}/{:} timesteps.".format(
            number_of_loaded_timesteps, idx_time))
        print("[utils_data] Loader restricted to {:}/{:} samples.".format(
            number_of_loaded_groups, idx))
        assert len(self.seq_paths) == number_of_loaded_groups

    def __getitem__(self, idx):
        return idx

    def getSequencesPart(self, idxs, t0, tend):
        data = []
        for idx in idxs:
            data.append(self.getSequencePart(idx, t0, tend))
        data = np.array(data)
        return data

    def getSequencePart(self, idx, t0, tend, scale=True):
        gname_seq = self.seq_paths[idx]["gname_seq"]
        group_seq = self.h5_file[gname_seq]

        timesteps = self.seq_paths[idx]["timesteps"]
        x = []
        assert t0 <= len(timesteps)
        assert tend <= len(
            timesteps
        ), "[getSequencePart()] Error: in data len(timesteps)={:}, while tend={:} requested. If testing, prediction horizon is too big, reduce it.".format(
            len(timesteps), tend)
        for timestep in timesteps[t0:tend]:
            group_time = group_seq[timestep]
            data_timestep = group_time["data"]
            x.append(data_timestep)
        x = np.array(x)

        if self.scaler is not None and scale:
            x = self.scaler.scaleData(x, single_sequence=True)

        return x


def getHDF5dataset(data_path, data_info_dict):
    if data_info_dict["structured"]:
        dataset = HDF5DatasetStructured(
            data_path,
            data_info_dict=data_info_dict,
        )
    else:
        dataset = HDF5Dataset(
            data_path,
            recursive=False,
            load_data=False,
            data_cache_size=3,
            data_info_dict=data_info_dict,
        )
    return dataset


def getDataLoader(
    data_path,
    data_info_dict,
    batch_size=1,
    shuffle=False,
    gpu=0,
):

    dataset = getHDF5dataset(data_path, data_info_dict)
    
    if gpu:
        generator = torch.Generator(device='cuda')
    else:
        generator = torch.Generator(device='cpu')
    data_loader = torch.utils.data.DataLoader(dataset,
                                              shuffle=shuffle,
                                              batch_size=batch_size,
                                              pin_memory=False,
                                              num_workers=0,
                                              generator=generator,
                                              drop_last=True)

    return data_loader, dataset


def getDataBatch(model, batch_of_sequences, start, stop, dataset=None):
    if model.data_info_dict["structured"]:
        data = dataset.getSequencesPart(batch_of_sequences, start, stop)
    else:
        data = batch_of_sequences[:, start:stop]
    return data


def saveData(data, data_path, protocol):

    assert (protocol in ["pickle"])
    saveDataPickle(data, data_path)

def saveDataPickle(data, data_path, add_file_format=True):
    
    if add_file_format: data_path += ".pickle"
    with open(data_path, "wb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        del data

def loadData(data_path, protocol, add_file_format=True):
    
    assert (protocol in ["hickle", "pickle"])
    return loadDataPickle(data_path, add_file_format=add_file_format)

def loadDataPickle(data_path, add_str="", add_file_format=True):
    
    if add_file_format: data_path += ".pickle"
    try:
        with open(data_path, "rb") as file:
            data = pickle.load(file)
    except Exception as inst:
        print("{:}[utils_data] Datafile\n {:s}\nNOT FOUND.".format(add_str, data_path))
        raise ValueError(inst)

    return data


def getDataHDF5Field(filename, field):
    
    with h5py.File(filename, "r") as f:
        assert field in f.keys(), "[utils_data] ERROR: Field {:} not in file {:}.".format(
            field, filename)
        data = np.array(f[field])
    
    return data