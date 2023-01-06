import torch
import numpy as np
from torch.utils.data import Dataset


class PNASDataset(Dataset):

    def __init__(self, file_path, mode='train'):
        super().__init__()

        # Search for txt files
        self.data = np.load(file_path+f'/{mode}.npz')['data'] # (N, 2, 3) or (N, 1, 3)

    # 0 --> 1
    def __getitem__(self, index):

        trace = self.data[index]

        input = trace[0]
        target = trace[0] if len(trace)==1 else trace[1]
        # target = trace[1] - trace[0] # diff

        input = torch.from_numpy(input[np.newaxis]).float()
        target = torch.from_numpy(target[np.newaxis]).float()

        return input, target

    def __len__(self):
        return len(self.data)