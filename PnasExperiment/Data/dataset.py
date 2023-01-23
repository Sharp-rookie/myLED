import torch
import numpy as np
from torch.utils.data import Dataset


class PNASDataset(Dataset):

    def __init__(self, file_path, mode='train', length=None):
        super().__init__()
        
        self.length = length

        # Search for txt files
        if length is None:
            self.data = np.load(file_path+f'/{mode}.npz')['data'] # (N, 2, 1, 3) or (N, 1, 1, 3)
        else:
            self.data = np.load(file_path+f'/{mode}_{length}.npz')['data']

    # 0 --> 1
    def __getitem__(self, index):

        trace = self.data[index]

        input = trace[0]
        if self.length is None:
            target = trace[0] if len(trace)==1 else trace[1]
        else:
            target = trace[self.length-1]

        input = torch.from_numpy(input).float() # (1, channel, feature_dim)
        target = torch.from_numpy(target).float()

        if self.length is None:
            return input.unsqueeze(0), target.unsqueeze(0)
        else:
            return input.unsqueeze(0), target.unsqueeze(0), [torch.from_numpy(trace[i]).float().unsqueeze(0) for i in range(self.length)] 

    def __len__(self):
        return len(self.data)
    

class PNASDataset4TCN(Dataset):

    def __init__(self, file_path, mode='train', length=None, sequence_length=None):
        super().__init__()
        
        assert length >= sequence_length
        self.sequence_length = sequence_length
        self.data = np.load(file_path+f'/{mode}_{length}.npz')['data'] # (trace_num, sequence_length, 1, 3)

    # 0 --> 1
    def __getitem__(self, index):

        trace = self.data[index]

        input = trace[:self.sequence_length-1]
        target = trace[-1]

        input = torch.from_numpy(input).float() # (sequence_length, channel, feature_dim)
        target = torch.from_numpy(target).float().unsqueeze(0) # (1, channel, feature_dim)

        return input, target

    def __len__(self):
        return len(self.data)