import torch

def weights_normal_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, mean=1.0, std=0.01)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(m.bias)

def weights_one_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.ones_(m.weight)
        if m.bias is not None:
            torch.nn.init.ones_(m.bias)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.ones_(m.weight)
        torch.nn.init.ones_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.ones_(m.weight)
        torch.nn.init.ones_(m.bias)
        
def weights_xavier_uniform_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)