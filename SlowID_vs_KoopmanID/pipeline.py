import torch

from model import Koopman_System, TIME_LAGGED_AE


def main():
    
    # params
    batch_size = 128
    koopman_dim = 8
    input_1d_width = 32
    
    # models
    system = Koopman_System(in_channels=1, input_1d_width=input_1d_width, embed_dim=64, koopman_dim=koopman_dim)
    esitimater = TIME_LAGGED_AE(in_channels=1, input_1d_width=input_1d_width, embed_dim=64)
    
    # generate koopman system observation traces
    for _ in range(1000):
        koopman_var = torch.Tensor([batch_size, koopman_dim])