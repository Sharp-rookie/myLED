import torch
from torch import nn


class Koopman_OPT(nn.Module):

    def __init__(self, koopman_dim):
        super(Koopman_OPT, self).__init__()

        self.koopman_dim = koopman_dim
        
        # (tau,) --> (koopman_dim, koopman_dim)
        self.V = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, self.koopman_dim**2),
            nn.Unflatten(-1, (self.koopman_dim, self.koopman_dim))
        )
        
        # (tau,) --> (koopman_dim,)
        self.Lambda = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, self.koopman_dim)
        )

    def forward(self, tau):

        # K: (koopman_dim, koopman_dim), K = V * Lambda * V^-1
        V, Lambda = self.V(tau), self.Lambda(tau)
        K = torch.mm(torch.mm(V, torch.diag(Lambda)), torch.inverse(V))

        return K


class Koopman_System(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, embed_dim, koopman_dim):
        super(Koopman_System, self).__init__()
        
        # (batchsize, koopman_dim) --> (batchsize, 1, in_channels, input_1d_width)
        self.decoder = nn.Sequential(
            nn.Linear(koopman_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
            nn.Tanh(),
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*input_1d_width, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (1, in_channels, input_1d_width))
        )
        
        self.K_opt = Koopman_OPT(koopman_dim)


class TIME_LAGGED_AE(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, embed_dim):
        super(TIME_LAGGED_AE, self).__init__()
                
        # (batchsize, 1, in_channels, input_1d_width) --> (batchsize, embed_dim)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*input_1d_width, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
            nn.Tanh(),
        )
        
        # (batchsize, embed_dim) --> (batchsize, 1, in_channels, input_1d_width)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*input_1d_width, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (1, in_channels, input_1d_width))
        )
        
        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, input_1d_width, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, input_1d_width, dtype=torch.float32))
        
    def forward(self,x):
        embed = self.encoder(x)
        out = self.decoder(embed)
        return out, embed        