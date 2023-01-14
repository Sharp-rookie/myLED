import torch
import numpy as np
from torch.utils.data import Dataset


class JCP12Dataset(Dataset):

    def __init__(self, file_path, mode='train', T=None, T_max=None):
        super().__init__()
        
        self.T = T

        # Search for txt files
        if T_max is None:
            self.data = np.load(file_path+f'/{mode}.npz')['data'] # (N, 2, 1, 4) or (N, 1, 1, 4)
        else:
            self.data = np.load(file_path+f'/{mode}_{T_max}.npz')['data']

    # 0 --> 1
    def __getitem__(self, index):

        trace = self.data[index]

        input = trace[0]
        if self.T is None:
            target = trace[0] if len(trace)==1 else trace[1]
        else:
            target = trace[self.T-1]

        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()

        return input, target # (1, channel, feature_dim)

    def __len__(self):
        return len(self.data)