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

    data = np.load('logs/2S2F-static/LearnDynamics/seed1/val/epoch-60/slow_vs_input.npz')
    slow_var = data['slow_vars']
    target = data['inputs'][...,:2].squeeze()

    # max_epoch = 10000
    # lr =0.001

    # net = Net(2,2)
    # optim = torch.optim.Adam(net.parameters(), lr=lr)
    # loss_fn = nn.L1Loss()

    # train_size = int(0.1*len(slow_var))
    # train_indices = random.sample(range(len(slow_var)), train_size)
    # test_indices = list(set(range(len(slow_var))) - set(train_indices))
    # train_x = torch.from_numpy(slow_var[train_indices])
    # train_y = torch.from_numpy(target[train_indices])
    # test_x = torch.from_numpy(slow_var[test_indices])
    # test_y = torch.from_numpy(target[test_indices])

    # for i in range(max_epoch):
    #     y = net(train_x)
    #     loss = loss_fn(y, train_y)

    #     optim.zero_grad()
    #     loss.backward()
    #     optim.step()

    #     print(f'\r[{i+1}/{max_epoch}] loss: {loss.data:.5f}', end='')

    # y = net(test_x)
    # print(f'val loss: {loss_fn(y, test_y):.5f}')

    # # plot slow variable vs input
    # plt.figure(figsize=(9,6))
    # plt.title('Val Reconstruction Curve')
    # for id_var in range(2):
    #     for index, item in enumerate([f'c{k}' for k in range(test_y.shape[-1])]):
    #         ax = plt.subplot(2, 2, index+1+2*(id_var))
    #         ax.scatter(test_y.detach().numpy()[:,index], y.detach().numpy()[:, id_var], s=1)
    #         ax.set_xlabel(item)
    #         ax.set_ylabel(f'U{id_var+1}')
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # plt.savefig(f"slow_vs_input.pdf", dpi=300)
    # plt.close()

    # plt.figure(figsize=(16,6))
    # for index, item in enumerate([f'c{k}' for k in range(test_y.shape[-1])]):
    #     ax = plt.subplot(1, 2, index+1)
    #     ax.plot(test_y.detach().numpy()[::10, index], label='true')
    #     ax.plot(y.detach().numpy()[::10, index], label='pred')
    #     ax.legend()
    # plt.subplots_adjust(wspace=0.35, hspace=0.35)
    # plt.savefig(f"recons.pdf", dpi=300)
    # plt.close()

    # plot recons in slow manifold
    import models
    model = models.DynamicsEvolver(in_channels=1, feature_dim=4, embed_dim=64, slow_dim=2, redundant_dim=2-2, tau_s=0.1, device='cuda:0', data_dim=4, enc_net='MLP', e1_layer_n=2)
    ckpt = torch.load('logs/2S2F-static/LearnDynamics/seed1/checkpoints/epoch-60.ckpt')
    model.load_state_dict(ckpt)
    model = model.to('cpu')

    num = int(5.1/0.01)
                    
    c1, c2 = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
    omega = 3
    c3 = np.sin(omega*c1)*np.sin(omega*c2)
    c4 = 1/((1+np.exp(-omega*c1))*(1+np.exp(-omega*c2)))
    
    fig = plt.figure(figsize=(16,6))
    recons = torch.as_tensor(data['inputs'])
    descale = model.descale(recons)
    for i, (c, trace) in enumerate(zip([c3,c4], [descale[:num,0,0,2:3],descale[:num,0,0,3:4]])):
        ax = plt.subplot(1,2,i+1,projection='3d')

        # plot the slow manifold and c3,c4 trajectory
        ax.scatter(c1, c2, c, marker='.', color='k', label=rf'Points on slow-manifold surface')
        ax.plot(descale[:num,0,0,:1], descale[:num,0,0,1:2], trace, linewidth=2, color="r", label=rf'Slow trajectory')
        ax.set_xlabel(r"$c_1$", labelpad=10, fontsize=18)
        ax.set_ylabel(r"$c_2$", labelpad=10, fontsize=18)
        ax.set_zlim(0, 2)
        ax.text2D(0.85, 0.65, rf"$c_{2+i+1}$", fontsize=18, transform=ax.transAxes)
        # ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(False)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.view_init(elev=20., azim=110.) # view direction: elve=vertical angle ,azim=horizontal angle
        # ax.view_init(elev=0., azim=-90.) # view direction: elve=vertical angle ,azim=horizontal angle
        plt.tick_params(labelsize=16)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.zaxis.set_major_locator(plt.MultipleLocator(1))
        if i == 1:
            plt.legend()
        plt.subplots_adjust(bottom=0., top=1.)
    
    plt.savefig(f"recons_manifold.pdf", dpi=300)
    plt.close()