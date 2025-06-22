'''
@File    :   cnn_dn_v1.py
@Time    :   2023/06/05 15:00
@Author  :   Hong Shijie
@Version :   1.3
@Note    :   在v1的基础上，适用于联合bp方案中的循环锁定
'''

import torch
from torch import nn
from utils.cnn.backbone_v1 import backbone
from utils.dn.dn_torch_v1 import DN
import torch.nn.functional as F


class concat_net(nn.Module):
    def __init__(self, batch_size, dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag):
        super(concat_net, self).__init__()
        self.backbone = backbone(z_neuron_num, batch_size)
        self.backbone.resnet18.fc = nn.Sequential()
        self.backbone = self.backbone.to('cuda:1')

        self.dn = DN(dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag).to('cuda:0')

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, z, mode, per_item, epo):
        x = self.backbone(x)
        if mode == 'lock_backbone':
            y_activated_num = self.dn(x.to('cuda:0'), z.to('cuda:0'), mode, per_item, epo)
            return y_activated_num
        elif mode == 'lock_dn':
            output = self.dn(x.to('cuda:0'), z.to('cuda:0'), mode, per_item, epo)
            return output
        elif mode == 'test':
            output = self.dn(x.to('cuda:0'), z.to('cuda:0'), mode, 1, epo)
            return output


