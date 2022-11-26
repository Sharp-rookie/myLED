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


class Cnov_AE(nn.Module):
    
    def __init__(self, in_channels, input_1d_width):
        super(Cnov_AE,self).__init__()

        # Input_padding_layer, (batchsize,2,101)-->(batchsize,2,128)
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

        # Conv_encoder_layer, (batchsize,2,128)-->(batchsize,64,1)
        self.conv_stack1 = nn.Sequential(
            conv1d_bn_relu(in_channels,32,4,stride=2),
            conv1d_bn_relu(32,32,3)
        )
        self.conv_stack2 = nn.Sequential(
            conv1d_bn_relu(32,32,4,stride=2),
            conv1d_bn_relu(32,32,3)
        )
        self.conv_stack3 = nn.Sequential(
            conv1d_bn_relu(32,64,4,stride=2),
            conv1d_bn_relu(64,64,3)
        )
        self.conv_stack4 = nn.Sequential(
            conv1d_bn_relu(64,64,4,stride=2),
            conv1d_bn_relu(64,64,3),
        )

        self.conv_stack5 = nn.Sequential(
            conv1d_bn_relu(64,64,4,stride=2),
            conv1d_bn_relu(64,64,3),
        )

        self.conv_stack6 = nn.Sequential(
            conv1d_bn_relu(64,64,4,stride=2),
            conv1d_bn_relu(64,64,3),
        )

        self.conv_stack7 = nn.Sequential(
            conv1d_bn_relu(64,64,4,stride=2),
            conv1d_bn_relu(64,64,3),
        )
        
        # Conv_time-lagged_decoder_layer,(batchsize,64,1)-->(batchsize,2,128)
        self.deconv_7 = deconv_tanh(64,64,4,stride=2)
        self.deconv_6 = deconv_tanh(67,64,4,stride=2)
        self.deconv_5 = deconv_tanh(67,64,4,stride=2)
        self.deconv_4 = deconv_tanh(67,64,4,stride=2)
        self.deconv_3 = deconv_tanh(67,32,4,stride=2)
        self.deconv_2 = deconv_tanh(35,16,4,stride=2)
        self.deconv_1 = deconv_tanh(19,in_channels,4,stride=2)
        
        # Conv_time-lagged_predictor_layer,(batchsize,64,1)-->(batchsize,2,128)
        self.predict_7 = nn.Conv1d(64,3,3,stride=1,padding=1)
        self.predict_6 = nn.Conv1d(67,3,3,stride=1,padding=1)
        self.predict_5 = nn.Conv1d(67,3,3,stride=1,padding=1)
        self.predict_4 = nn.Conv1d(67,3,3,stride=1,padding=1)
        self.predict_3 = nn.Conv1d(67,3,3,stride=1,padding=1)
        self.predict_2 = nn.Conv1d(35,3,3,stride=1,padding=1)
        self.up_sample_7 = nn.Sequential(
            nn.ConvTranspose1d(3,3,4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )
        self.up_sample_6 = nn.Sequential(
            nn.ConvTranspose1d(3,3,4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )
        self.up_sample_5 = nn.Sequential(
            nn.ConvTranspose1d(3,3,4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )
        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose1d(3,3,4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose1d(3,3,4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose1d(3,3,4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )

        # Output_unpadding_layer, (batchsize,2,128)-->(batchsize,2,101)
        self.output_unpad = Unpad(padding_input)

    def encoder(self, x):
        
        x = self.input_pad(x)
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        conv6_out = self.conv_stack6(conv5_out)
        conv7_out = self.conv_stack7(conv6_out)
        return conv7_out

    def decoder(self, x):

        deconv7_out = self.deconv_7(x)
        predict_7_out = self.up_sample_7(self.predict_7(x))

        concat_6 = torch.cat([deconv7_out,predict_7_out],dim=1)
        deconv6_out = self.deconv_6(concat_6)
        predict_6_out = self.up_sample_6(self.predict_6(concat_6))

        concat_5 = torch.cat([deconv6_out,predict_6_out],dim=1)
        deconv5_out = self.deconv_5(concat_5)
        predict_5_out = self.up_sample_5(self.predict_5(concat_5))

        concat_4 = torch.cat([deconv5_out,predict_5_out],dim=1)
        deconv4_out = self.deconv_4(concat_4)
        predict_4_out = self.up_sample_4(self.predict_4(concat_4))

        concat_3 = torch.cat([deconv4_out,predict_4_out],dim=1)
        deconv3_out = self.deconv_3(concat_3)
        predict_3_out = self.up_sample_3(self.predict_3(concat_3))

        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        predict_out = self.deconv_1(concat1)

        predict_out = self.output_unpad(predict_out)
        
        return predict_out

    def forward(self,x, reconstructed_latent):
        
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent


if __name__ == '__main__':
    ae = Cnov_AE(2, 101)
    input = torch.ones((128, 2, 101))
    latent = ae.encoder(input)
    output = ae.decoder(latent)
    print(output.shape)