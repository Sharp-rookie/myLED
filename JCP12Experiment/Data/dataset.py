import torch
import numpy as np
from torch.utils.data import Dataset


class JCP12Dataset(Dataset):

    def __init__(self, file_path, mode='train', length=None, neural_ode=False):
        super().__init__()
        
        self.length = length
        self.neural_ode = neural_ode

        # Search for txt files
        if self.neural_ode:
            self.data = np.load(file_path+f'/neural_ode_{mode}.npz')['data'] # (trace_num, sample_num, 2, 1, 4)
        else:
            self.data = np.load(file_path+f'/{mode}.npz')['data'] # (trace_num*sample_num, 2, 1, 4)


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
            input =  torch.from_numpy(sample[:,:1]).float()
            target =  torch.from_numpy(sample[:,1:]).float()
            return input, target

    def __len__(self):
        return len(self.data) if not self.neural_ode else len(self.data[0])
    
#     def plot(self):
        
#         inputs = []
#         targets = []
#         for i in range(self.__len__()):
#             input, target, _ = self.__getitem__(i)
#             inputs.append(input[0,0,2])
#             targets.append(target[0,0,2])
#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.plot(inputs[:100], label='input')
#         plt.plot(targets[:100], label='target')
#         plt.legend()
#         plt.savefig('data.pdf', dpi=300)

# data_path = 'Data/data/tau_' + str(0.15)
# j = JCP12Dataset(data_path, 'val', length=10)
# j.plot()