import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
import matplotlib.pyplot as plt


def init_normal(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0, std=0.01)
        nn.init.zeros_(layer.bias)


class Net(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, output_dim)
        )

        self.apply(init_normal)

    def forward(self, x):

        return self.net(x)
    

if __name__ == '__main__':

    data = np.load('logs/2S2F-static/LearnDynamics/seed1/val/epoch-100/slow_vs_input.npz')
    slow_var = data['slow_vars']
    target = data['inputs'][...,:2].squeeze()

    max_epoch = 10000
    lr =0.001

    net = Net(2,2)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    train_size = int(0.1*len(slow_var))
    train_indices = random.sample(range(len(slow_var)), train_size)
    test_indices = list(set(range(len(slow_var))) - set(train_indices))
    train_x = torch.from_numpy(slow_var[train_indices])
    train_y = torch.from_numpy(target[train_indices])
    test_x = torch.from_numpy(slow_var[test_indices])
    test_y = torch.from_numpy(target[test_indices])

    for i in range(max_epoch):
        y = net(train_x)
        loss = loss_fn(y, train_y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f'\r[{i+1}/{max_epoch}] loss: {loss.data:.5f}', end='')

    y = net(test_x)
    print(f'val loss: {loss_fn(y, test_y):.5f}')

    # plot slow variable vs input
    plt.figure(figsize=(9,6))
    plt.title('Val Reconstruction Curve')
    for id_var in range(2):
        for index, item in enumerate([f'c{k}' for k in range(test_y.shape[-1])]):
            ax = plt.subplot(2, 2, index+1+2*(id_var))
            ax.scatter(test_y.detach().numpy()[:,index], y.detach().numpy()[:, id_var], s=1)
            ax.set_xlabel(item)
            ax.set_ylabel(f'U{id_var+1}')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig(f"slow_vs_input.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(16,6))
    for index, item in enumerate([f'c{k}' for k in range(test_y.shape[-1])]):
        ax = plt.subplot(1, 2, index+1)
        ax.plot(test_y.detach().numpy()[::10, index], label='true')
        ax.plot(y.detach().numpy()[::10, index], label='pred')
        ax.legend()
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    plt.savefig(f"recons.pdf", dpi=300)
    plt.close()