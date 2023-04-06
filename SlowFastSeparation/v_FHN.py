import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layer):
        super(Net, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        import ipdb;ipdb.set_trace()
        h0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out


def fhn(y, t):
    
    a, b, I, epsilon = 0.7, 0.8, 0.5, 0.08

    u, v = y
    dudt = u - u**3/3 - v + I
    dvdt = epsilon * (u + a - b*v)

    return [dudt, dvdt]
    

def simulation(u_bound, v_bound, t, n_trace, u_step, v_step, dir, u_max, u_min, v_max, v_min):

    os.makedirs(dir, exist_ok=True)
    
    trace = []
    for i in range(n_trace):

        # 在给定区域内随机选取初始值
        u0, v0 = np.random.uniform(u_bound, u_bound+u_step), np.random.uniform(v_bound, v_bound+v_step)
        u0, v0 = np.clip(u0, u_bound, u_bound+u_step), np.clip(v0, v_bound, v_bound+v_step)
        y0 = [u0, v0]

        # 运行微分方程求解器，获得模拟结果
        sol = odeint(fhn, y0=y0, t=t)

        trace.append(sol)
    
    trace = np.array(trace)

    # 保存模拟结果
    np.savez(f'{dir}/origin_{n_trace}.npz', trace=trace, u_bound=u_bound, v_bound=v_bound, u_step=u_step, v_step=v_step)

    plt.figure(figsize=(10,12))
    for i in range(2):
        ax = plt.subplot(2,1,i+1)
        for j in range(n_trace):
            ax.plot(t, trace[j, :, i], c='b', alpha=0.05, label='trajectory' if j==0 else None)
        ax.set_xlabel('Time / s')
        ax.set_ylabel(['u', 'v'][i])
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f'{dir}/fhn.jpg', dpi=300)
    plt.close()
    
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111)
    for j in range(n_trace):
        sol = trace[j]
        ax.plot(sol[:, 1], sol[:, 0], c='b', alpha=0.05, label='trajectory' if j==0 else None)

    u1 = np.linspace(u_min, u_max, 500)
    v1 = u1 - u1**3/3 + 0.5
    v2 = np.linspace(v_min, v_max, 500)
    u2 = 0.8*v2 - 0.7
    ax.plot(v1, u1, c='r', label='u-nullcline (slow manifold)')
    ax.plot(v2, u2, c='g', label='v-nullcline(dv/dt=0)')
    ax.set_xlabel('v (slow)')
    ax.set_ylabel('u (fast)')
    ax.legend(loc='upper right')
    ax.text(0.2, 0.5, r'$\frac{du}{dt} > 0$', transform=ax.transAxes, fontsize=14)
    ax.text(0.8, 0.6, r'$\frac{dv}{dt} < 0$', transform=ax.transAxes, fontsize=14)
    # 绘制初始范围框
    ax.add_patch(plt.Rectangle((v_bound, u_bound), v_step, u_step, fill=False, edgecolor='k', lw=1))
    # 绘制全局所有初始范围框
    for _u_bound in np.arange(u_min, u_max, u_step):
        for _v_bound in np.arange(v_min, v_max, v_step):
            ax.add_patch(plt.Rectangle((_v_bound, _u_bound), v_step, u_step, fill=False, edgecolor='k', lw=0.5, alpha=0.5))
    ax.set_xlim(v_min-0.1, v_max+0.1)
    ax.set_ylim(u_min-0.1, u_max+0.1)
    ax.set_title(f'{n_trace} trajectories simulated for {int(np.max(t))}s')
    plt.savefig(f'{dir}/phase.jpg', dpi=300)
    plt.close()

    return trace


def train():

    try:
        data = np.load('v_fhn/origin_100.npz')['trace']
    except:
        simdata = simulation(
            u_bound=-2.0,
            v_bound=-0.5,
            t=np.arange(0., 1000., 0.001),
            n_trace=100,
            u_step=4.0,
            v_step=2.0,
            dir='v_fhn',
            u_max=2.0,
            u_min=-2.0,
            v_max=1.5,
            v_min=-0.5
            )
        data = np.load('v_fhn/origin_100.npz')['trace']
    
    train_data = data[:80]
    test_data = data[80:]

    train_x = train_data[:, :-1, 1].reshape(-1, 1, 1)  # 
    train_y = train_data[:, 1:, 1].reshape(-1, 1)
    test_x = test_data[:, :-1, :].reshape(-1, 2)
    test_y = test_data[:, 1:,:].reshape(-1, 2)

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = Net(input_size=1, hidden_size=64, output_size=2, n_layer=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'epoch: {epoch}, loss: {loss.item()}')
    
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(test_x).to(device)).cpu().numpy()
        plt.figure(figsize=(10,12))
        for i in range(2):
            ax = plt.subplot(2,1,i+1)
            ax.plot(test_y[:, i], c='b', label='ground truth')
            ax.plot(pred[:, i], c='r', label='prediction')
            ax.set_xlabel('Time / s')
            ax.set_ylabel(['u', 'v'][i])
            ax.legend()
        plt.subplots_adjust(hspace=0.3)
        plt.savefig('v_fhn/prediction.jpg', dpi=300)
        plt.close()
    

if __name__ == '__main__':

    train()