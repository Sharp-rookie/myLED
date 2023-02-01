import torch
import numpy as np
from torch.utils.data import Dataset


class PNASDataset(Dataset):

    def __init__(self, file_path, mode='train', length=None, neural_ode=False):
        super().__init__()
        
        self.length = length
        self.neural_ode = neural_ode

        # Search for txt files
        if self.neural_ode:
            self.data = np.load(file_path+f'/neural_ode_{mode}.npz')['data'] # (1*sample_num, 1, 1, 3)
        else:
            self.data = np.load(file_path+f'/{mode}.npz')['data'] # (trace_num*sample_num, 2, 1, 3)

    # 0 --> 1
    def __getitem__(self, index):

        if not self.neural_ode:
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

        else:
            sample = self.data[:, index]
            # input =  torch.from_numpy(sample[:,:1]).float()
            # target =  torch.from_numpy(sample[:,1:]).float()
            input =  torch.from_numpy(sample).float()
            target =  input
            return input.unsqueeze(-2), target.unsqueeze(-2)
        
    def __len__(self):
        return len(self.data) if not self.neural_ode else len(self.data[0])