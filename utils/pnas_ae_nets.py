import torch
import numpy as np
from torch import nn


def isPowerOfTwo(n):
    return (n & (n - 1) == 0) and n != 0

def findNextPowerOfTwo(n):
    while (not isPowerOfTwo(n)):
        n += 1
    return n

class Unpad(nn.Module):
    def __init__(self, padding):
        super(Unpad, self).__init__()
        self.padding = padding
        assert (np.shape(list(padding))[0] == 2)

    def forward(self, x):
        if self.padding[1] > 0:
            return x[..., self.padding[0]:-self.padding[1]]
        elif self.padding[1] == 0:
            return x[..., self.padding[0]:]
        else:
            raise ValueError("Not implemented.")

def conv1d_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = nn.Sequential(
        nn.Conv1d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        nn.BatchNorm1d(outch),
        nn.ReLU()
    )
    return convlayer

def deconv_tanh(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = nn.Sequential(
        nn.ConvTranspose1d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        nn.BatchNorm1d(outch),
        nn.Tanh()
    )
    return convlayer

def mlp_relu_drop(input_dim, output_dim):
    mlplayer = nn.Sequential(
        nn.Linear(input_dim, output_dim, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.01)
    )
    return mlplayer

class Cnov_AE(nn.Module):
    
    def __init__(self, in_channels, input_1d_width):
        super(Cnov_AE,self).__init__()

        # Input_padding_layer, (batchsize,1,3)-->(batchsize,1,4)
        if not isPowerOfTwo(input_1d_width):
            # Add padding in the first layer to make power of two
            input_1d_width_padded = findNextPowerOfTwo(input_1d_width)
            padding = input_1d_width_padded - input_1d_width
            if padding % 2 == 0:
                pad_x_left = int(padding / 2)
                pad_x_right = int(padding / 2)
            else:
                pad_x_left = int((padding - 1) / 2)
                pad_x_right = int((padding - 1) / 2 + 1)
            padding_input = tuple([pad_x_left, pad_x_right])
        else:
            input_1d_width_padded = input_1d_width
            padding_input = tuple([0, 0])
        self.input_pad = nn.ConstantPad1d(padding_input, 0.0)

        # Conv_encoder_layer, (batchsize,1,4)-->(batchsize,64,1)
        self.conv_stack1 = nn.Sequential(
            conv1d_bn_relu(in_channels,32,4,stride=2),
            conv1d_bn_relu(32,32,3)
        )
        self.conv_stack2 = nn.Sequential(
            conv1d_bn_relu(32,32,4,stride=2),
            conv1d_bn_relu(32,64,3)
        )
        
        # Conv_time-lagged_decoder_layer,(batchsize,64,1)-->(batchsize,1,4)
        self.deconv_2 = deconv_tanh(64,32,4,stride=2)
        self.deconv_1 = deconv_tanh(35,in_channels,4,stride=2)
        
        # Conv_time-lagged_predictor_layer,(batchsize,64,1)-->(batchsize,1,4)
        self.predict_2 = nn.Conv1d(64,3,3,stride=1,padding=1)
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose1d(3,3,4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )

        # Output_unpadding_layer, (batchsize,1,4)-->(batchsize,1,3)
        self.output_unpad = Unpad(padding_input)

    def encoder(self, x):

        x = self.input_pad(x)
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        return conv2_out

    def decoder(self, x):

        deconv2_out = self.deconv_2(x)
        predict_2_out = self.up_sample_2(self.predict_2(x))

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        predict_out = self.deconv_1(concat1)

        predict_out = self.output_unpad(predict_out)
        
        return predict_out

    def forward(self,x):
        
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent


class MLP_AE(nn.Module):
    
    def __init__(self, in_channels, input_1d_width):
        super(MLP_AE,self).__init__()

        # Input_padding_layer, (batchsize,2*101)-->(batchsize,256)
        if not isPowerOfTwo(in_channels*input_1d_width):
            # Add padding in the first layer to make power of two
            input_1d_width_padded = findNextPowerOfTwo(in_channels*input_1d_width)
            padding = input_1d_width_padded - in_channels*input_1d_width
            if padding % 2 == 0:
                pad_x_left = int(padding / 2)
                pad_x_right = int(padding / 2)
            else:
                pad_x_left = int((padding - 1) / 2)
                pad_x_right = int((padding - 1) / 2 + 1)
            padding_input = tuple([pad_x_left, pad_x_right])
        else:
            input_1d_width_padded = in_channels*input_1d_width
            padding_input = tuple([0, 0])
        self.input_pad = nn.ConstantPad1d(padding_input, 0.0)

        # MLP_encoder_layer, (batchsize,2,256)-->(batchsize,64,1)
        self.num_layers = 1
        self.hidden_size = 256
        self.en_lstm = nn.LSTM(input_size=input_1d_width_padded, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.mlp_stack = mlp_relu_drop(256, 64)
        
        # Conv_time-lagged_decoder_layer,(batchsize,64,1)-->(batchsize,256)
        self.demlp1 = mlp_relu_drop(64, 256)
        self.demlp2 = mlp_relu_drop(256, 256)
        self.demlp3 = mlp_relu_drop(256, input_1d_width_padded)

        # Output_unpadding_layer, (batchsize,256)-->(batchsize,2*101)
        self.output_unpad = Unpad(padding_input)

    def encoder(self, x):

        x = nn.Flatten()(x)
        x = self.input_pad(x).unsqueeze(-2)

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        lstm_out, (h_out, _) = self.en_lstm(x, (h_0, c_0))
        lstm_out = lstm_out.view(-1, self.hidden_size)

        mlp_out = self.mlp_stack(lstm_out)
        mlp_out = mlp_out.unsqueeze(-1)

        return mlp_out

    def decoder(self, x):

        x = x.squeeze()
        x = self.demlp1(x)
        x = self.demlp2(x)
        x = self.demlp3(x)
        predict_out = self.output_unpad(x)
        predict_out = nn.Unflatten(-1, (2, 101))(predict_out)
        
        return predict_out

    def forward(self,x):
        
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent


if __name__ == '__main__':
    ae = Cnov_AE(1, 3)
    input = torch.ones((128, 1, 3))
    latent = ae.encoder(input)
    print(latent.shape, end='')
    output = ae.decoder(latent)
    print(output.shape)