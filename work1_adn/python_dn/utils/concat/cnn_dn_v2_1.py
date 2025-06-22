'''
@File    :   cnn_dn_v2.py
@Time    :   2023/05/25 19:00
@Author  :   Hong Shijie
@Version :   2.1
@Note    :   将resnet提取出的结果分别输入fc和dn 并将其经过softmax，相加后的结果进行loss计算，而后只对fc进行反向传播
'''

import torch
from torch import nn
from utils.cnn.backbone_v2_1 import backbone
from utils.dn.dn_torch_v7 import DN
import torch.nn.functional as F


class concat_net(nn.Module):
    def __init__(self, batch_size, dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag):
        super(concat_net, self).__init__()
        self.backbone = backbone(z_neuron_num, batch_size)

        self.dn = DN(dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # for name, parameter in self.dn.named_parameters():
        #     parameter.requires_grad = False
        #     print(name, parameter)

    def forward(self, x, z, mode, per_item):
        x, output_fc = self.backbone(x)
        if mode == 'train':
            output_dn, activated_num = self.dn(x, z, mode, per_item)
            output_fc = F.softmax(output_fc, dim=1)
            output_dn = torch.where(output_dn < 0.5, -1.0, output_dn)
            output = output_fc + output_dn
            # output = output_fc.mul(output_dn)
            # output = F.one_hot(output, num_classes=self.dn.z_neuron_num)
            # output = output.float().unsqueeze(0)
            return output, activated_num
        elif mode == 'test':
            output_dn = self.dn(x, z, mode, 1)
            output_fc = F.softmax(output_fc, dim=1)
            output = output_fc + output_dn
            # output = output_fc.mul(output_dn)
            return output_fc, output_dn, output

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


