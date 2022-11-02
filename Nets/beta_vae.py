import torch
import torch.nn as nn


# latent_state_dim
class Beta_Vae(nn.Module):
    def __init__(self, embedding_size=5):
        super(Beta_Vae, self).__init__()
        self.fc_mu = nn.Linear(embedding_size, embedding_size)
        self.fc_std = nn.Linear(embedding_size, embedding_size)

    def reparameterization_trick(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, latent_state):
        mu = self.fc_mu(latent_state)
        log_var = self.fc_std(latent_state)
        z = self.reparameterization_trick(mu, log_var)
        return z, mu, log_var