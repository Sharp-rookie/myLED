import torch
from torch import nn


class TIME_LAGGED_AE(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, embed_dim):
        super(TIME_LAGGED_AE, self).__init__()
                
        # (batchsize,1,3)-->(batchsize, embed_dim)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*input_1d_width, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
            nn.Tanh(),
        )
        
        # (batchsize, embed_dim)-->(batchsize,1,3)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, 3, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (in_channels, int(input_1d_width/in_channels)))
        )
        
        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, input_1d_width, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, input_1d_width, dtype=torch.float32))
        
    def forward(self,x):
        embed = self.encoder(x)
        out = self.decoder(embed)
        return out, embed

    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min