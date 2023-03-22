import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt


device = 'cuda:1'

# 加载数据集
train_data = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

net = Net().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练模型
for epoch in range(50):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
    print(f'\r[{epoch+1}/{50}] loss={loss.data:.5f}', end='')

train_features = net(train_data.data.float().to(device)).detach().cpu().numpy()

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2)
train_features_tsne = tsne.fit_transform(train_features)
train_labels = train_data.targets.numpy()
plt.scatter(train_features_tsne[:, 0], train_features_tsne[:, 1], c=train_labels)
plt.savefig('t-SNE.jpg', dpi=300)

# 使用 MDS 进行降维
tsne = MDS(n_components=2)
train_features_tsne = tsne.fit_transform(train_features)
train_labels = train_data.targets.numpy()
plt.scatter(train_features_tsne[:, 0], train_features_tsne[:, 1], c=train_labels)
plt.savefig('MDS.jpg', dpi=300)
