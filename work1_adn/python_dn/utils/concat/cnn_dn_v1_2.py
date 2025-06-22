'''
@File    :   cnn_dn_v1_2.py
@Time    :   2023/05/29 20:00
@Author  :   Hong Shijie
@Version :   1.2
@Note    :   在v1.1的基础上，适用于z神经元学习率与训练轮次有关的方案
'''

import torch
from torch import nn
from utils.cnn.backbone_v1 import backbone
from utils.dn.dn_torch_v6_11 import DN
import torch.nn.functional as F

class concat_net(nn.Module):
    def __init__(self, batch_size, dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag):
        super(concat_net, self).__init__()
        self.backbone = backbone(z_neuron_num, batch_size)
        self.backbone.resnet18.fc = nn.Sequential()

        self.dn = DN(dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # for name, parameter in self.dn.named_parameters():
        #     parameter.requires_grad = False
        #     print(name, parameter)

    def forward(self, x, z, mode, per_item, global_step):
        x = self.backbone(x)
        if mode == 'train':
            activated_num = self.dn(x, z, mode, per_item, global_step)
            # output = F.one_hot(output, num_classes=self.dn.z_neuron_num)
            # output = output.float().unsqueeze(0)
            return activated_num
        elif mode == 'test':
            output = self.dn(x, z, mode, 1, global_step)
            return output

    # def save(self):


    # def __getstate__(self):
    #     return {
    #         'backbone': self.backbone.__getstate__(),
    #         'dn': self.dn.__getstate__()
    #     }
    #
    # def __setstate__(self, state):
    #     self.backbone.__setstate__(state['backbone'])
    #     self.dn.__setstate__(state['dn'])


