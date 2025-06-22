'''
@File    :   cnn_dn_v3.py
@Time    :   2023/07/25 15:30
@Author  :   Hong Shijie
@Version :   3temp
@Note    :   暂时用于grl设置
'''
import torch
from torch import nn
from utils.cnn.backbone_v3_2 import ResNet18
from utils.dn.dn_torch_v6_11_1 import DN
import torch.nn.functional as F

class concat_net(nn.Module):
    def __init__(self, batch_size, dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag):
        super(concat_net, self).__init__()
        self.backbone = ResNet18(z_neuron_num)

        self.dn = DN(dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # for name, parameter in self.dn.named_parameters():
        #     parameter.requires_grad = False
        #     print(name, parameter)

    def forward(self, x, z, mode, per_item, global_step):
        x, features = self.backbone(x)
        if mode == 'train':
            activated_num = self.dn(features, z, mode, per_item, global_step)
            # output = F.one_hot(output, num_classes=self.dn.z_neuron_num)
            # output = output.float().unsqueeze(0)
            return activated_num
        elif mode == 'test':
            output = self.dn(features, z, mode, 1, global_step)
            return output
