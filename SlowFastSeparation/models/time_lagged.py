import torch
from torch import nn

from .weight_init import normal_init


class EncodeCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, flatten=False):
        super(EncodeCell, self).__init__()

        self.flatten = nn.Flatten() if flatten else None

        self.w1 = nn.Parameter(torch.rand(input_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.rand(input_dim, hidden_dim))
        self.w3 = nn.Parameter(torch.rand(input_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.rand(hidden_dim))
        self.b2 = nn.Parameter(torch.rand(hidden_dim))
        self.b3 = nn.Parameter(torch.rand(hidden_dim))
        self.b4 = nn.Parameter(torch.rand(hidden_dim))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x):

        if self.flatten:
            x = self.flatten(x)

        # # GRU1
        # p1 = self.sigmoid(x @ self.w1 + self.b1)
        # p2 = self.tanh(x @ self.w2 + self.b2)
        # y = (1 - p1) * p2

        # GRU2
        p1 = self.sigmoid(x @ self.w1 + self.b1)
        p2 = self.tanh(x @ self.w3 + self.b3 + self.sigmoid(x@self.w2+self.b2) * self.b4)
        y = (1 - p1) * p2

        # # LSTM
        # p1 = self.sigmoid(x @ self.w1 + self.b1)
        # p2 = self.sigmoid(x @ self.w2 + self.b2)
        # p3 = self.tanh(x @ self.w3 + self.b3)
        # y = p1 * self.tanh(p2 * p3)
        # # y = (1-p1) * self.tanh(p2 * p3)

        return y


class TimeLaggedAE(nn.Module):
    
    def __init__(self, in_channels, feature_dim, embed_dim, data_dim):
        super(TimeLaggedAE, self).__init__()
                
        # # (batchsize,1,1,4)-->(batchsize, embed_dim)
        # self.encoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_channels*feature_dim, 64, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.01),
        #     nn.Linear(64, embed_dim, bias=True),
        #     nn.Tanh(),
        # )
        # self.layer_num = 3
        # self.hidden_dim = embed_dim
        # # self.encoder = nn.LSTM(
        # #     input_size=in_channels*feature_dim, 
        # #     hidden_size=self.hidden_dim, 
        # #     num_layers=self.layer_num, 
        # #     dropout=0.01, 
        # #     batch_first=True # input: (batch_size, squences, features)
        # #     )
        # self.encoder = nn.GRU(
        #     input_size=in_channels*feature_dim, 
        #     hidden_size=self.hidden_dim, 
        #     num_layers=self.layer_num, 
        #     dropout=0.01, 
        #     batch_first=True # input: (batch_size, squences, features)
        #     )
        self.encoder = nn.Sequential(
            EncodeCell(in_channels*feature_dim, 64, True),
            EncodeCell(64, 64),
            EncodeCell(64, embed_dim),
        )
        
        # (batchsize, embed_dim)-->(batchsize,1,1,4)
        self.decoder_prior = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*feature_dim, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (1, in_channels, feature_dim))
        )

        # (batchsize, embed_dim)-->(batchsize,1,1,4)
        self.decoder_reverse = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*feature_dim, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (1, in_channels, feature_dim))
        )
        
        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, data_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, data_dim, dtype=torch.float32))

        # init
        self.apply(normal_init)
        
    def forward(self, x, direct='prior'):
        embed = self.encoder(x)

        # h0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device='cpu')
        # c0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device='cpu')
        # x = nn.Flatten(start_dim=2)(x)
        # _, (h, c)  = self.encoder(x, (h0, c0))
        # _, h  = self.encoder(x, h0)
        # embed = h[-1]
        
        out = self.decoder_prior(embed) if direct=='prior' else self.decoder_reverse(embed)
        return out, embed

    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min