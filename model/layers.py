import torch
import torch.nn as nn
import torch.nn.init as init

import cvnn.init as cinit

class CLinear(nn.Module):
    def __init__(self, in_features, out_features, enable_bias):
        super(CLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.enable_bias = enable_bias
        # Until to PyTorch 1.9.0, it still does not support torch.ComplexFloatTensor and torch.ComplexDoubleTensor.
        self.weight = nn.Parameter(torch.complex(torch.FloatTensor(in_features, out_features), \
                        torch.FloatTensor(in_features, out_features)))
        if self.enable_bias == True:
            self.bias = nn.Parameter(torch.complex(torch.FloatTensor(out_features), torch.FloatTensor(out_features)))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        cinit.complex_xavier_uniform_(self.weight)
        #cinit.complex_kaiming_uniform_(self.weight)

        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x):
        c_linear = torch.mm(x, self.weight)
        
        if self.bias is not None:
            return c_linear + self.bias
        else:
            return c_linear