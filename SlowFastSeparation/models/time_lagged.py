import torch
from torch import nn

from .weight_init import normal_init


class EncodeCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, flatten=False, dropout=False, net='MLP', act=True):
        super(EncodeCell, self).__init__()

        assert net in ['MLP', 'GRU1', 'GRU2', 'LSTM', 'LSTM-changed'], f"Encoder Net Error, {net} not implemented!"

        self.net = net

        self.act = act
        self.flatten = nn.Flatten() if flatten else None
        self.dropout = nn.Dropout(p=0.01) if dropout else None

        self.w1 = nn.Parameter(torch.normal(mean=0., std=0.01, size=(input_dim, hidden_dim)))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        if net != 'MLP': # GRU1/2 or LSTM
            self.w2 = nn.Parameter(torch.normal(mean=0., std=0.01, size=(input_dim, hidden_dim)))
            self.b2 = nn.Parameter(torch.zeros(hidden_dim))

        if net != 'GRU1': # GRU2 or LSTM
            self.w3 = nn.Parameter(torch.normal(mean=0., std=0.01, size=(input_dim, hidden_dim)))
            self.b3 = nn.Parameter(torch.zeros(hidden_dim))

        if 'LSTM' not in net: # GRU2
            self.b4 = nn.Parameter(torch.zeros(hidden_dim))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x):

        if self.flatten:
            x = self.flatten(x)
        
        if self.net == 'MLP':
            if self.act:
                y = self.tanh(x @ self.w1 + self.b1)
            else:
                y = x @ self.w1 + self.b1
        elif self.net == 'GRU1':
            z = self.sigmoid(x @ self.w1 + self.b1)
            h = self.tanh(x @ self.w2 + self.b2)
            y = (1 - z) * h
        elif self.net == 'GRU2':
            z = self.sigmoid(x @ self.w1 + self.b1)
            r = self.sigmoid(x @ self.w2 + self.b2)
            h = self.tanh(x @ self.w3 + self.b3 + r * self.b4)
            y = (1 - z) * h
        elif self.net == 'LSTM':
            O = self.sigmoid(x @ self.w1 + self.b1)
            I = self.sigmoid(x @ self.w2 + self.b2)
            C = self.tanh(x @ self.w3 + self.b3)
            y = O * self.tanh(I * C)
        elif self.net == 'LSTM-changed':
            O = self.sigmoid(x @ self.w1 + self.b1)
            I = self.sigmoid(x @ self.w2 + self.b2)
            C = self.tanh(x @ self.w3 + self.b3)
            y = (1-O) * self.tanh(I * C)

        if self.dropout:
            y = self.dropout(y)

        return y


class TimeLaggedAE(nn.Module):
    
    def __init__(self, in_channels, feature_dim, embed_dim, data_dim, enc_net='MLP', e1_layer_n=3):
        super(TimeLaggedAE, self).__init__()
                
        # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)
        assert e1_layer_n>1, f"Layer num of encoder_1 must larger than 1, but is {e1_layer_n}"
        self.encoder = nn.Sequential(EncodeCell(in_channels*feature_dim, 64, True, True, enc_net))
        [self.encoder.append(EncodeCell(64, 64, False, True, enc_net)) for _ in range(e1_layer_n-2)]
        self.encoder.append(EncodeCell(64, embed_dim, False, False, enc_net, False))
        # self.encoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_channels*feature_dim, 64, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.01),
        #     nn.Linear(64, embed_dim, bias=True),
        # )
        
        # (batchsize, embed_dim)-->(batchsize,1,channel_num,feature_dim)
        self.decoder_prior = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*feature_dim, bias=True),
            nn.Unflatten(-1, (1, in_channels, feature_dim))
        )

        # (batchsize, embed_dim)-->(batchsize,1,channel_num,feature_dim)
        self.decoder_reverse = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*feature_dim, bias=True),
            nn.Unflatten(-1, (1, in_channels, feature_dim))
        )
        
        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, data_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, data_dim, dtype=torch.float32))

        # init
        self.apply(normal_init)
        
    def forward(self, x, direct='prior'):
        embed = self.encoder(x)
        out = self.decoder_prior(embed) if direct=='prior' else self.decoder_reverse(embed)
        return out, embed
    
    def enc(self, x):
        return self.encoder(x)
    
    def dec(self, embed, direct='prior'):
        return self.decoder_prior(embed) if direct=='prior' else self.decoder_reverse(embed)

    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min