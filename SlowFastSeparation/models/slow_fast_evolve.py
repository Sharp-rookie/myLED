import torch
from torch import nn

from .weight_init import normal_init
from .time_lagged import EncodeCell


class Koopman_OPT(nn.Module):

    def __init__(self, koopman_dim):
        super(Koopman_OPT, self).__init__()

        self.koopman_dim = koopman_dim

        self.register_parameter('V', nn.Parameter(torch.Tensor(koopman_dim, koopman_dim)))
        self.register_parameter('Lambda', nn.Parameter(torch.Tensor(koopman_dim)))

        # init
        torch.nn.init.normal_(self.V, mean=0, std=0.1)
        torch.nn.init.normal_(self.Lambda, mean=0, std=0.1)

    def forward(self, tau):

        # K: (koopman_dim, koopman_dim), K = V * exp(tau*Lambda) * V^-1
        tmp1 = torch.diag(torch.exp(tau*self.Lambda))
        V_inverse = torch.inverse(self.V)
        K = torch.mm(torch.mm(V_inverse, tmp1), self.V)

        return K
    

class LSTM_OPT(nn.Module):
    
    def __init__(self, in_channels, feature_dim, hidden_dim, layer_num, tau_s, device):
        super(LSTM_OPT, self).__init__()
        
        self.device = device
        self.tau_s = tau_s
        
        # (batchsize,1,channel_num,feature_dim)-->(batchsize,1,channel_num*feature_dim)
        self.flatten = nn.Flatten(start_dim=2)
        
        # (batchsize,1,feature_dim)-->(batchsize, hidden_dim)
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(
            input_size=in_channels*feature_dim, 
            hidden_size=hidden_dim, 
            num_layers=layer_num, 
            dropout=0.01, 
            batch_first=True # input: (batch_size, squences, features)
            )
        
        # (batchsize, hidden_dim)-->(batchsize, feature_dim)
        self.fc = nn.Linear(hidden_dim, in_channels*feature_dim)

        # mechanism new
        self.a_head = nn.Linear(hidden_dim, 1)
        self.b_head = nn.Linear(hidden_dim, in_channels*feature_dim)
        
        # (batchsize, 4)-->(batchsize,1,channel_num,feature_dim)
        self.unflatten = nn.Unflatten(-1, (in_channels, feature_dim))
    
    def forward(self, x):
        
        h0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=self.device)
        c0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=self.device)
        
        x = self.flatten(x)
        _, (h, c)  = self.cell(x, (h0, c0))

        # y = self.fc(h[-1])

        a = self.a_head(h[-1]).unsqueeze(-2) # (batch_size, 1, 1)
        b = self.b_head(h[-1]).unsqueeze(-2) # (batch_size, 1, in_channels*feature_dim)
        y = torch.abs(x) * torch.exp(-(a+self.tau_s)) * b # (batch_size, 1, in_channels*feature_dim)

        y = self.unflatten(y)
                
        return y


class DynamicsEvolver(nn.Module):
    
    def __init__(self, in_channels, feature_dim, embed_dim, slow_dim, redundant_dim, tau_s, device, data_dim, enc_net='MLP'):
        super(DynamicsEvolver, self).__init__()

        self.slow_dim = slow_dim
        
        # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)
        assert enc_net in ['MLP', 'GRU1', 'GRU2', 'LSTM', 'LSTM-changed'], f"Encoder Net Error, {enc_net} not implemented!"
        self.encoder_1 = nn.Sequential(
            EncodeCell(in_channels*feature_dim, 64, True, True, enc_net),
            EncodeCell(64, 64, False, True, enc_net),
            EncodeCell(64, embed_dim, False, False, enc_net, False),
        )
        # self.encoder_1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_channels*feature_dim, 64, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.01),
        #     nn.Linear(64, embed_dim, bias=True),
        # )
        
        # (batchsize, embed_dim)-->(batchsize, slow_dim)
        self.encoder_2 = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, slow_dim, bias=True),
        )
        # self.encoder_2_1 = nn.Sequential(
        #     nn.Linear(embed_dim, 64, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.01),
        #     nn.Linear(64, 1, bias=True),
        # )
        # self.encoder_2_2 = nn.Sequential(
        #     nn.Linear(embed_dim, 64, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.01),
        #     nn.Linear(64, 1, bias=True),
        # )

        # (batchsize, slow_dim)-->(batchsize, redundant_dim)
        self.encoder_3 = nn.Sequential(
            nn.Linear(slow_dim, 32, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(32, redundant_dim, bias=True),
        )

        # (batchsize, slow_dim)-->(batchsize, embed_dim)
        self.decoder_1 = nn.Sequential(
            nn.Linear(slow_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
        )
        # self.decoder_1_1 = nn.Sequential(
        #     nn.Linear(1, 64, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.01),
        #     nn.Linear(64, embed_dim, bias=True),
        # )
        # self.decoder_1_2 = nn.Sequential(
        #     nn.Linear(1, 64, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.01),
        #     nn.Linear(64, embed_dim, bias=True),
        # )

        # (batchsize, embed_dim)-->(batchsize,1,channel_num,feature_dim)
        self.decoder_2 = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*feature_dim, bias=True),
            nn.Unflatten(-1, (1, in_channels, feature_dim))
        )
        # self.decoder_2_1 = nn.Sequential(
        #     nn.Linear(embed_dim, 64, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.01),
        #     nn.Linear(64, in_channels*feature_dim, bias=True),
        #     nn.Unflatten(-1, (1, in_channels, feature_dim))
        # )
        # self.decoder_2_2 = nn.Sequential(
        #     nn.Linear(embed_dim, 64, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.01),
        #     nn.Linear(64, in_channels*feature_dim, bias=True),
        #     nn.Unflatten(-1, (1, in_channels, feature_dim))
        # )
        
        self.K_opt = Koopman_OPT(slow_dim+redundant_dim)
        self.lstm = LSTM_OPT(in_channels, feature_dim, hidden_dim=64, layer_num=2, tau_s=tau_s, device=device)

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, data_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, data_dim, dtype=torch.float32))

        # init
        self.apply(normal_init)
    
    def obs2slow(self, obs):
        # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)-->(batchsize, slow_dim)
        embed = self.encoder_1(obs)
        slow_var = self.encoder_2(embed)
        
        return slow_var, embed
    # def obs2slow_new(self, obs):
    #     # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)-->(batchsize, slow_dim)
    #     embed = self.encoder_1(obs)
    #     slow_var1 = self.encoder_2_1(embed)
    #     slow_var2 = self.encoder_2_2(embed)
        
    #     return slow_var1, slow_var2, embed
    # def obs2slow_new2(self, obs):
    #     # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)-->(batchsize, slow_dim)
    #     embed = self.encoder_1(obs)

    #     from sklearn.manifold import TSNE
    #     from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
    #     pca = KernelPCA(n_components=4, kernel='cosine')
    #     svd = TruncatedSVD(n_components=4)
    #     tsne = TSNE(n_components=4)
    #     slow_var = pca.fit_transform(embed.detach().cpu().numpy())
    #     slow_var = torch.from_numpy(slow_var).to(embed.device)
        
    #     return slow_var, embed

    def slow2obs(self, slow_var):
        # (batchsize, slow_dim)-->(batchsize,1,channel_num,feature_dim)
        embed = self.decoder_1(slow_var)
        obs = self.decoder_2(embed.detach())
        # obs = self.decoder_2(embed)

        return obs, embed
    # def slow2obs_new2(self, slow_var):
    #     # (batchsize, slow_dim)-->(batchsize,1,channel_num,feature_dim)
    #     embed = self.decoder_1(slow_var)
    #     obs1 = self.decoder_2_1(embed.detach())
    #     obs2 = self.decoder_2_2(embed.detach())
    #     # obs = self.decoder_2(embed)

        # return obs1, obs2, embed
    # def slow2obs_new(self, slow_var1, slow_var2):
    #     # (batchsize, slow_dim)-->(batchsize,1,channel_num,feature_dim)
    #     embed1 = self.decoder_1_1(slow_var1)
    #     embed2 = self.decoder_1_2(slow_var2)
    #     obs = self.decoder_2((embed1+embed2).detach())
    #     # obs = self.decoder_2(embed1+embed2)

    #     return obs, embed1, embed2

    def slow2koopman(self, slow_var):
        # (batchsize, slow_dim)-->(batchsize, koopman_dim = slow_dim + redundant_dim)
        redundant_var = self.encoder_3(slow_var)
        koopman_var = torch.concat((slow_var, redundant_var), dim=-1)
        return koopman_var

    def koopman2slow(self, koopman_var):
        # (batchsize, koopman_dim = slow_dim + redundant_dim)-->(batchsize, slow_dim)
        slow_var = koopman_var[:,:self.slow_dim]
        return slow_var

    def koopman_evolve(self, koopman_var, tau=1.):
        
        K = self.K_opt(tau)
        koopman_var_next = torch.matmul(K, koopman_var.unsqueeze(-1)).squeeze(-1)
        
        return koopman_var_next
    
    def lstm_evolve(self, x, T=1):
        
        y = [self.lstm(x)]
        for _ in range(1, T-1): 
            y.append(self.lstm(y[-1]))
            
        return y[-1], y[:-1]
    
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min