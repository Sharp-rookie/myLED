import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sdeint import itoSRI2, itoEuler


def weight_init(m):
    if hasattr(m, 'weight'):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, horizon, output_size, n_layer):
        super(RNN, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size, n_layer, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, n_layer, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon*output_size)

        self.apply(weight_init)
    
    def forward(self, x):
        h0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size).to(x.device)
        # out, h = self.rnn(x, h0)
        out, h = self.gru(x, h0)
        # out, (h, c) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1])
        out = out.view(-1, self.horizon, self.output_size)
        return out


class MLP(nn.Module):
    def __init__(self, lookback, input_size, hidden_size, horizon, output_size):
        super(MLP, self).__init__()
        self.horizon = horizon
        self.output_size = output_size

        self.fc1 = nn.Linear(lookback*input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, horizon*output_size)
        self.relu = nn.ReLU()

        self.apply(weight_init)
    
    def forward(self, x):
        x = nn.Flatten()(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.view(-1, self.horizon, self.output_size)
        return out


class SDE_FHN():
    def __init__(self, a, epsilon, delta1, delta2):
        self.a = a
        self.epsilon = epsilon
        self.delta1 = delta1
        self.delta2 = delta2
    
    def f(self, x, t):
        return np.array([(x[0]-1/3*x[0]**3-x[1])/self.epsilon, 
                         x[0]+self.a])
    
    def g(self, x, t):
        return np.array([[self.delta1*1., 0.], 
                        [0., self.delta2*1.]])
    

def simulation(u0, v0, a, epsilon, delta1, delta2, t, dir, n_trace):

    os.makedirs(dir, exist_ok=True)

    sde = SDE_FHN(a=a, epsilon=epsilon, delta1=delta1, delta2=delta2)
    tspan = np.arange(0, t, 0.001)
    trace = np.zeros((n_trace, len(tspan), 2))
    for i in range(n_trace):
        sol = itoSRI2(sde.f, sde.g, [u0, v0], tspan)
        trace[i, :, :] = sol

    # 保存模拟结果
    np.savez(f'{dir}/origin_{n_trace}_delta2_{delta2}.npz', trace=trace)

    # 绘制膜电位和恢复变量随时间演化的图形
    plt.figure(figsize=(10,12))
    for i in range(2):
        ax = plt.subplot(2,1,i+1)
        # for j in range(n_trace):
        #     ax.plot(tspan, trace[j, :, i], c='b', alpha=0.05, label='trajectory' if j==0 else None)
        ax.plot(tspan, trace[0, :, i], c='b', label='trajectory')
        ax.set_xlabel('Time / s')
        ax.set_ylabel(['u', 'v'][i])
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f'{dir}/fhn_delta2_{delta2}.jpg', dpi=300)
    plt.close()
                
    # 绘制相空间演化轨迹
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111)
    # 绘制多条相空间演化轨迹
    # for j in range(n_trace):
    #     ax.plot(trace[j, :, 1], trace[j, :, 0], c='b', alpha=0.05, label='trajectory' if j==0 else None)
    ax.plot(trace[0, :, 1], trace[0, :, 0], c='b', label='trajectory')
    # 绘制u、v的nullcline
    u1 = np.linspace(-4.0, 4.0, 200)
    v1 = u1-u1**3/3
    v2 = np.linspace(-2.5, 2.5, 200)
    u2 = np.ones_like(v2)*(-a)
    ax.plot(v1, u1, c='r', label='u-nullcline (slow manifold))')
    ax.plot(v2, u2, c='g', label='v-nullcline(dv/dt=0)')
    ax.set_xlabel('v (slow)')
    ax.set_ylabel('u (fast)')
    ax.legend(loc='upper right')
    # 注明dudt、dvdt的正负
    ax.text(0.2, 0.7, r'$\frac{du}{dt} > 0$', transform=ax.transAxes, fontsize=14)
    ax.text(0.15, 0.3, r'$\frac{dv}{dt} < 0$', transform=ax.transAxes, fontsize=14)
    # 设置x、y轴范围
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-4.0, 4.0)
    plt.savefig(f'{dir}/phase_delta2_{delta2}.jpg', dpi=300)
    plt.close()

    return trace


def train():

    n_trace = 10
    delta2 = 0.1
    dir = 'v_fhn_pans17'

    print('load data...')
    try:
        data = np.load(f'{dir}/origin_{n_trace}_delta2_{delta2}.npz')['trace']
    except:
        simdata = simulation(
            a=1.05,
            epsilon=0.01,
            delta1=0.2,
            delta2=delta2,
            u0=-2.,
            v0=0.5,
            t=50.,
            dir=dir,
            n_trace=n_trace
            )
        data = np.load(f'{dir}/origin_{n_trace}_delta2_{delta2}.npz')['trace']
    print('over')
    
    data = data.astype(np.float32)
    train_data = data[:7] # 70%的数据用于训练, shape=(70, 50000, 2)
    test_data = data[7:] # 30%的数据用于测试, shape=(30, 50000, 2)

    # 数据预处理
    lookback = 10  # 和NMI22用的相同
    horizon = 20
    window_size = lookback + horizon
    train_data = np.array([train_data[:, i:i+window_size, :] for i in range(train_data.shape[1]-window_size)]).reshape(-1, window_size, 2)
    test_data = np.array([test_data[:, i:i+window_size, :] for i in range(test_data.shape[1]-window_size)]).reshape(-1, window_size, 2)
    train_x = train_data[:, :lookback, 0].reshape(-1, lookback, 1)  # (batch, lookback, input_size)
    test_x = test_data[:, :lookback, 0].reshape(-1, lookback, 1)
    train_y = train_data[:, lookback:, :].reshape(-1, horizon, 2)  # (batch, horizon, input_size)
    test_y = test_data[:, lookback:, :].reshape(-1, horizon, 2)
    
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = RNN(input_size=1, hidden_size=32, horizon=horizon, output_size=2, n_layer=1).to(device)
    # model = MLP(lookback=lookback, input_size=1, hidden_size=32, horizon=horizon, output_size=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_epoch = 5

    try:
        model.load_state_dict(torch.load(f'{dir}/model_{max_epoch}.pth'))
    except:
        for epoch in range(1, max_epoch+1):
            model.train()
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            if epoch % 1 == 0:
                print(f'\repoch: {epoch}, loss: {loss.item():.6f}', end='')
            if epoch % 50 == 0:
                torch.save(model.state_dict(), f'{dir}/model_{epoch}.pth')
    
    model.eval()
    with torch.no_grad():
        input = torch.from_numpy(test_x).to(device)
        pred = model(input)
        pred = pred.cpu().numpy()
        pred = pred.reshape(-1, 2)
        test_y = test_y.reshape(-1, 2)
    mse = np.mean((pred - test_y)**2)
    print(f'\nmse: {mse:.6f}')

    plt.figure(figsize=(21, 6))
    plt.subplot(131)
    plt.plot(test_y[::3, 0], test_y[::3, 1], 'b.', label='true')
    plt.plot(pred[::3, 0], pred[::3, 1], 'r.', label='pred')
    plt.xlabel('v')
    plt.ylabel('u')
    plt.legend()
    plt.subplot(132)
    plt.plot(test_y[::3, 0], 'b.', label='true')
    plt.plot(pred[::3, 0], 'r.', label='pred')
    plt.xlabel('time')
    plt.ylabel('u')
    plt.legend()
    plt.subplot(133)
    plt.plot(test_y[::3, 1], 'b.', label='true')
    plt.plot(pred[::3, 1], 'r.', label='pred')
    plt.xlabel('time')
    plt.ylabel('v')
    plt.legend()
    plt.savefig(f'{dir}/pred_{max_epoch}.jpg', dpi=300)
    plt.close()
    

if __name__ == '__main__':

    train()