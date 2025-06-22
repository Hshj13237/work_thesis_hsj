'''
@File    :   cnn_dn_v1_2_1.py
@Time    :   2023/06/20 12:00
@Author  :   Hong Shijie
@Version :   1.2.1
@Note    :   在v1.2的基础上，适应原始图像参与y神经元激活值计算方案
'''

import torch
from torch import nn
from utils.cnn.backbone_v1 import backbone
from utils.dn.dn_torch_v8 import DN
import torch.nn.functional as F
import torchvision

class concat_net(nn.Module):
    def __init__(self, batch_size, img_dim, dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent,
                 y_top_down_percent, y_img_percent, z_neuron_num, synapse_flag):
        super(concat_net, self).__init__()
        self.backbone = backbone(z_neuron_num, batch_size)
        self.backbone.resnet18.fc = nn.Sequential()

        self.dn = DN(img_dim, dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent,
                     y_top_down_percent, y_img_percent, z_neuron_num, synapse_flag)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # for name, parameter in self.dn.named_parameters():
        #     parameter.requires_grad = False
        #     print(name, parameter)

    def forward(self, x, z, mode, per_item, global_step):
        img = x.clone()
        img = torchvision.transforms.functional.rgb_to_grayscale(img)
        x = self.backbone(x)
        if mode == 'train':
            activated_num = self.dn(img, x, z, mode, per_item, global_step)
            # output = F.one_hot(output, num_classes=self.dn.z_neuron_num)
            # output = output.float().unsqueeze(0)
            return activated_num
        elif mode == 'test':
            output = self.dn(img, x, z, mode, 1, global_step)
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


