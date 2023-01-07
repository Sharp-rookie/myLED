import torch
import numpy as np
from torch.utils.data import Dataset


class PNASDataset(Dataset):

    def __init__(self, file_path, mode='train', T=None):
        super().__init__()
        
        self.T = T

        # Search for txt files
        if T is None:
            self.data = np.load(file_path+f'/{mode}.npz')['data'] # (N, 2, 3) or (N, 1, 3)
        else:
            self.data = np.load(file_path+f'/{mode}_pred.npz')['data']

    # 0 --> 1
    def __getitem__(self, index):

        trace = self.data[index]

        input = trace[0]
        if self.T is None:
            target = trace[0] if len(trace)==1 else trace[1]
        else:
            target = trace[self.T-1]

        input = torch.from_numpy(input[np.newaxis]).float()
        target = torch.from_numpy(target[np.newaxis]).float()

        return input, target

    def __len__(self):
        return len(self.data)