# -*- coding: utf-8 -*-
import torch
from torch import nn


class Koopman_OPT(nn.Module):

    def __init__(self, koopman_dim, delta_t):
        super(Koopman_OPT, self).__init__()

        self.koopman_dim = koopman_dim
        self.num_eigenvalues = int(koopman_dim/2)
        self.delta_t = delta_t
        self.parameterization = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.koopman_dim, self.num_eigenvalues*2),
            nn.Tanh(),
            nn.Linear(self.num_eigenvalues*2, self.num_eigenvalues*2)
        )

    def forward(self, x):
        
        # x: (B, koopman_dim)
        batch_size = x.shape[0]
        
        # mu: (B, koopman_dim/2), omega: (B, koopman_dim/2)
        mu, omega = torch.unbind(self.parameterization(x).reshape(-1, self.num_eigenvalues, 2), -1)

        # K: (B, koopman_dim, koopman_dim)
        K = torch.zeros(batch_size, self.koopman_dim, self.koopman_dim, dtype=torch.float32)
        exp = torch.exp(self.delta_t * mu)
        cos = torch.cos(self.delta_t * omega)
        sin = torch.sin(self.delta_t * omega)
        for i in range(0, self.koopman_dim, 2):
            index = i//2
            K[:, i + 0, i + 0] = cos[:,index] *  exp[:,index]
            K[:, i + 0, i + 1] = -sin[:,index] * exp[:,index]
            K[:, i + 1, i + 0] = sin[:,index]  * exp[:,index]
            K[:, i + 1, i + 1] = cos[:,index] * exp[:,index]

        # y = K * x
        y = torch.matmul(K, x.unsqueeze(-1)).squeeze()

        return y
    

class LSTM_OPT(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, hidden_dim, layer_num):
        super(LSTM_OPT, self).__init__()
        
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(
            input_size=in_channels*input_1d_width, 
            hidden_size=hidden_dim, 
            num_layers=layer_num, 
            dropout=0.01, 
            batch_first=True # input: (batch_size, squences, features)
            )
        
        self.fc = nn.Linear(hidden_dim, in_channels*input_1d_width)
        self.unflatten = nn.Unflatten(-1, (in_channels, int(input_1d_width/in_channels)))
    
    def forward(self, x):
        
        h0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32)
        c0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32)
        _, hc  = self.cell(x, (h0, c0))
        y = self.fc(hc[0][-1])
        y = self.unflatten(y)
                
        return y


class EVOLVER(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, embed_dim, slow_dim, delta_t):
        super(EVOLVER, self).__init__()
        
        self.encoder_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*input_1d_width, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
            nn.Tanh(),
        )
        
        self.encoder_2 = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, slow_dim, bias=True),
            nn.Tanh(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(slow_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
            nn.Tanh(),
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*input_1d_width, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (in_channels, input_1d_width))
        )
        
        self.K_opt = Koopman_OPT(slow_dim, delta_t)
        self.delta_t = delta_t
        
        self.lstm = LSTM_OPT(in_channels, input_1d_width, hidden_dim=64, layer_num=2)

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, input_1d_width, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, input_1d_width, dtype=torch.float32))

    def extract(self, x):
        embed = self.encoder_1(x)
        slow_var = self.encoder_2(embed)
        return slow_var, embed

    def recover(self, x):
        x = self.decoder(x)
        return x

    def koopman_evolve(self, x, T=1):
        
        y = self.K_opt(x)
        for _ in range(1, T): 
            y = self.K_opt(y)
        return y
    
    def lstm_evolve(self, x, T=1):
        
        y = self.lstm(x)
        for _ in range(1, T): 
            y = self.lstm(y)
        return y
    
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min