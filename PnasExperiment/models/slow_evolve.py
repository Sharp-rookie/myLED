# -*- coding: utf-8 -*-
import torch
from torch import nn


class K_OPT(nn.Module):

    def __init__(self, koopman_dim, delta_t):
        super(K_OPT, self).__init__()

        self.koopman_dim = koopman_dim
        self.num_eigenvalues = int(koopman_dim/2)
        self.delta_t = delta_t
        self.parameterization = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.koopman_dim, self.num_eigenvalues*2),
            nn.Tanh(),
            nn.Linear(self.num_eigenvalues*2, self.num_eigenvalues*2)
        )

    def forward(self,x,T):
        # x is (B, koopman_dim)
        # mu is (B, koopman_dim/2)
        # omega is (B, koopman_dim/2)

        batch_size = x.shape[0]
        Y = torch.Variable(torch.zeros(batch_size, T, self.koopman_dim))
        y = x[:, 0, :]
        
        for t in range(T):
            mu, omega = torch.unbind(self.parameterization(y).reshape(-1, self.num_eigenvalues, 2),-1)

            # K is (B, koopman_dim, koopman_dim)
            K = torch.Variable(torch.zeros(batch_size, self.koopman_dim, self.koopman_dim))
            exp = torch.exp(self.delta_t * mu)
            cos = torch.cos(self.delta_t * omega)
            sin = torch.sin(self.delta_t * omega)
            for i in range(0, self.koopman_dim, 2):
                index = i//2
                K[:, i + 0, i + 0] = cos[:,index] *  exp[:,index]
                K[:, i + 0, i + 1] = -sin[:,index] * exp[:,index]
                K[:, i + 1, i + 0] = sin[:,index]  * exp[:,index]
                K[:, i + 1, i + 1] = cos[:,index] * exp[:,index]

            y = torch.matmul(K, y.unsqueeze(-1)).squeeze(-1)
            Y[:, t, :] = y

        return Y


class SLOW_EVOLVER(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, embed_dim, slow_dim, delta_t):
        super(SLOW_EVOLVER, self).__init__()
        
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
            nn.Linear(64, 3, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (in_channels, int(input_1d_width/in_channels)))
        )
        
        self.K_opt = K_OPT(slow_dim, delta_t)
        self.delta_t = delta_t

        # scale inside the model
        self.register_buffer('min', torch.zeros((in_channels, input_1d_width)))
        self.register_buffer('max', torch.ones((in_channels, input_1d_width)))

    def extract(self,x):
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        return x

    def recover(self,x):
        x = self.decoder(x)
        return x

    def koopman_evolve(self,x,T=1):
        return self.K_opt(x, T)
    
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min