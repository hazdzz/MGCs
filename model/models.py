import torch
import torch.nn as nn
import torch.nn.functional as F

from model import layers
from cvnn import activation as act
from cvnn import dropout as drop

# The Real-Valued Simplifying Magnetic Graph Convolution Network
# for g = 0
class RSMGC(nn.Module):
    def __init__(self, n_feat, n_class, enable_bias):
        super(RSMGC, self).__init__()
        self.linear = nn.Linear(in_features=n_feat, out_features=n_class, bias=enable_bias)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.log_softmax(x)

        return x

# The Complex-Valued Simplifying Magnetic Graph Convolution Network
# for g in (0, 0.5]
class CSMGC(nn.Module):
    def __init__(self, n_feat, n_class, enable_bias):
        super(CSMGC, self).__init__()
        self.c_linear = layers.CLinear(in_features=n_feat, out_features=n_class, enable_bias=enable_bias)
        self.mod_log_softmax = act.modLogSoftmax(dim=1)

    def forward(self, x):
        x = self.c_linear(x)
        x = self.mod_log_softmax(x)

        return x

# The Real-Valued Magnetic Graph Convolution Network
# for g = 0
class RMGC(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, act_func, droprate):
        super(RMGC, self).__init__()
        self.linear_1 = nn.Linear(in_features=n_feat, out_features=n_hid, bias=enable_bias)
        self.linear_2 = nn.Linear(in_features=n_hid, out_features=n_class, bias=enable_bias)
        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'leaky_relu':
            self.act_func = nn.LeakyReLU()
        else:
            raise ImportError(f'ERROR: {act_func} is not imported for this model.')
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act_func(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.log_softmax(x)

        return x

# The Complex-Valued Magnetic Graph Convolution Network
# for g in (0, 0.5]
class CMGC(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, act_func, droprate):
        super(CMGC, self).__init__()
        self.c_linear_1 = layers.CLinear(in_features=n_feat, out_features=n_hid, enable_bias=enable_bias)
        self.c_linear_2 = layers.CLinear(in_features=n_hid, out_features=n_class, enable_bias=enable_bias)
        if act_func == 'c_relu':
            self.act_func = act.CReLU()
        elif act_func == 'z_relu':
            self.act_func = act.zReLU()
        elif act_func == 'mod_relu':
            self.act_func = act.modReLU()
        elif act_func == 'c_leaky_relu':
            self.act_func = act.CLeakyReLU()
        else:
            raise NotImplementedError(f'ERROR: {act_func} is an activation function for real-valued neural networks, \
                                        or this activation function is not implemented for the complex-valued neural \
                                        networks yet.')
        self.dropout = drop.Dropout(p=droprate)
        self.mod_log_softmax = act.modLogSoftmax(dim=1)

    def forward(self, x):
        x = self.c_linear_1(x)
        x = self.act_func(x)
        x = self.dropout(x)
        x = self.c_linear_2(x)
        x = self.mod_log_softmax(x)

        return x

# This model is proposed for testing boundary condition of the over-smoothing issue.
class LGC(nn.Module):
    def __init__(self, n_feat, n_class, enable_bias):
        super(LGC, self).__init__()
        self.linear = nn.Linear(in_features=n_feat, out_features=n_class, bias=enable_bias)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.log_softmax(x)

        return x