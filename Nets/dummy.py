import torch.nn as nn

class viewEliminateChannels(nn.Module):

    def __init__(self, params):
        super(viewEliminateChannels, self).__init__()
        self.Dx = params["Dx"]
        self.Dy = params["Dy"]
        self.Dz = params["Dz"]
        self.channels = params["channels"]
        self.input_dim = params["input_dim"]

        if self.channels == 1:
            self.shape_ = (self.input_dim, self.Dx)
            self.total_dim = self.input_dim * self.Dx
        elif self.channels == 2:
            self.shape_ = (self.input_dim, self.Dx, self.Dy)
            self.total_dim = self.input_dim * self.Dx * self.Dy
        elif self.channels == 3:
            self.shape_ = (self.input_dim, self.Dx, self.Dy, self.Dz)
            self.total_dim = self.input_dim * self.Dx * self.Dy * self.Dz
        else:
            raise ValueError("Not implemented.")

    def forward(self, x):

        shape_ = (x.size(0), self.total_dim)
        temp = x.view(shape_)
        return temp


class viewAddChannels(nn.Module):
    def __init__(self, params):
        super(viewAddChannels, self).__init__()
        self.Dx = params["Dx"]
        self.Dy = params["Dy"]
        self.Dz = params["Dz"]
        self.channels = params["channels"]
        self.input_dim = params["input_dim"]

        if self.channels == 1:
            self.shape_ = (self.input_dim, self.Dx)
            self.total_dim = self.input_dim * self.Dx
        elif self.channels == 2:
            self.shape_ = (self.input_dim, self.Dx, self.Dy)
            self.total_dim = self.input_dim * self.Dx * self.Dy
        elif self.channels == 3:
            self.shape_ = (self.input_dim, self.Dx, self.Dy, self.Dz)
            self.total_dim = self.input_dim * self.Dx * self.Dy * self.Dz
        else:
            raise ValueError("Not implemented.")

    def forward(self, x):
        shape_ = (x.size(0), *self.shape_)
        temp = x.view(shape_)
        return temp