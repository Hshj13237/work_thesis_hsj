import torch
from torch import nn

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

    def forward(self, x, y):
        s = x * y
        return s



