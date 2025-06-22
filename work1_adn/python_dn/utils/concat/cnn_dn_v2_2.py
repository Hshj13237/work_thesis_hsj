'''
@File    :   cnn_dn_v2.py
@Time    :   2023/06/27 19:00
@Author  :   Hong Shijie
@Version :   2.2
@Note    :
'''

import torch
from torch import nn
from utils.cnn.backbone_v2_2 import backbone
from utils.dn.dn_torch_v7_2 import DN
import torch.nn.functional as F


class concat_net(nn.Module):
    def __init__(self, batch_size, dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag):
        super(concat_net, self).__init__()
        self.backbone = backbone(batch_size)

        self.dn = DN(dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fc = nn.Linear(self.backbone.resnet18.fc.in_features, z_neuron_num)
        self.backbone.resnet18.fc = nn.Sequential()

        # for name, parameter in self.dn.named_parameters():
        #     parameter.requires_grad = False
        #     print(name, parameter)

    def forward(self, x, z, mode, per_item, global_step):
        x = self.backbone(x)
        if mode == 'train':
            with torch.no_grad():
                att_weight, activated_num = self.dn(x, z, mode, per_item, global_step)
            new_x = x.mul(att_weight)
            output = self.fc(new_x)
            return output, activated_num
        elif mode == 'test':
            output_dn, att_weight = self.dn(x, z, mode, 1, global_step)
            new_x = x.mul(att_weight)
            output = self.fc(new_x)
            return output_dn, output

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


