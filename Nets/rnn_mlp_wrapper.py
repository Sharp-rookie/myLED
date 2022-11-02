import torch.nn as nn

from .activations import getActivation


class RNN_MLP_wrapper(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, act, act_output):
        super(RNN_MLP_wrapper, self).__init__()

        self.hidden_sizes = [input_size] + hidden_sizes + [output_size]
        self.activation = getActivation(act)
        self.activation_output = getActivation(act_output)
        self.layers = nn.ModuleList()

        for ln in range(len(self.hidden_sizes) - 1):
            self.layers.append(nn.Linear(self.hidden_sizes[ln], self.hidden_sizes[ln + 1], bias=True))
            if ln < len(self.hidden_sizes) - 2:
                self.layers.append(self.activation)
            else:
                self.layers.append(self.activation_output)

    def forward(self, input_, state, is_train=False):
        for layer in self.layers:
            output = layer(input_)
            input_ = output
        return output, state
