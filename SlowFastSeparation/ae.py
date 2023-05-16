import torch
import torch.nn as nn
import numpy as np


class AE(nn.Module):
    def __init__(self, in_features, id, out_features):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, id)
        )
        self.decoder = nn.Sequential(
            nn.Linear(id, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        y = self.decoder(embedding)
        return embedding, y

id = 1
xdim = 1
clone = 1
data = np.load(f'logs_0.01s/PNAS17_xdim{xdim}_clone{clone}_delta0.2_du0.0-sliding_window-circle/TimeSelection/tau_0.0/seed1/test/epoch-200/embedding.npy')
train_num = int(0.5*data.shape[0])
train_data = torch.from_numpy(data[:train_num])
train_dataset = torch.utils.data.TensorDataset(train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
test_data = torch.from_numpy(data[train_num:])
test_dataset = torch.utils.data.TensorDataset(test_data)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

max_epoch = 50
net = AE(128, id, 128)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()

for i in range(max_epoch):
    for x in train_loader:
        x = torch.stack(x, dim=0)[0]
        embedding, y = net(x)
        loss = loss_func(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'[{i}/{max_epoch}] loss={loss.item():.5f}')