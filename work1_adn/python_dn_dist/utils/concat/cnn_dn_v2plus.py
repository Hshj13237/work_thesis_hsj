'''
@File    :   cnn_dn_v2plus.py
@Time    :   2024/04/24 15:00
@Author  :   Hong Shijie
@Version :   2
@Note    :   加入softloss和mseloss
'''

import torch
from torch import nn
from utils.cnn.backbone_v2 import ResNet18
from utils.dn.dn_torch_v1plus import DN
import torch.nn.functional as F
from torch.nn import Linear
from torchvision import transforms

class add_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y):
        y2fc = y
        y2dn = y
        return y2fc, y2dn

    @staticmethod
    def backward(ctx, grad_fc, grad_dn):
        grad_add = grad_dn + grad_fc
        return grad_add

class concat_net(nn.Module):
    def __init__(self, batch_size, dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag):
        super(concat_net, self).__init__()
        self.backbone = ResNet18(z_neuron_num)
        self.backbone = self.backbone.to('cuda:1')

        self.fc = Linear(in_features=512, out_features=z_neuron_num, bias=True).to('cuda:1')

        self.dn = DN(dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag).to('cuda:0')

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.add = add_grad.apply

        self.gray_trans = transforms.Grayscale(1)

        self.crossentropy = SoftTargetCrossEntropy()

    def forward(self, x, z, mode, per_item, epo):
        # train resnet+fc
        # x = self.backbone(x)
        # x = self.fc(x)
        # return x

        # img = self.gray_trans(x.data).squeeze(1)
        x = self.backbone(x)

        if mode == 'lock_backbone':
            y_activated_num = self.dn(x.to('cuda:0'), z.to('cuda:0'), mode, per_item, epo)
            return y_activated_num
        elif mode == 'lock_dn':
            x_fc, x_dn = self.add(x)
            output_dn, var_loss = self.dn(x_dn.to('cuda:0'), z.to('cuda:0'), mode, per_item, epo)
            output_fc = self.fc(x_fc)
            output_fc_max, _ = torch.max(output_fc, dim=1)
            output_fc_min, _ = torch.min(output_fc, dim=1)
            output = output_fc.to("cuda:0") + output_dn * (output_fc_max - output_fc_min).unsqueeze(1).to("cuda:0")
            loss = self.crossentropy(output, z.to('cuda:0')) + var_loss
            return loss
        # elif mode == 'lock_dn':
        #     output = self.dn(x.to('cuda:0'), z.to('cuda:0'), mode, per_item, epo)
        #     return output
        elif mode == 'test':
            output_dn = self.dn(x.to('cuda:0'), z.to('cuda:0'), mode, 1, epo)
            output_fc = self.fc(x)

            output_dn_max, _ = torch.max(output_dn, dim=1)

            ch_idx = torch.nonzero(output_dn_max.data < 0.1).squeeze(1)
            output_fc_max, _ = torch.max(output_fc, dim=1)
            output_fc_min, _ = torch.min(output_fc, dim=1)
            output_fc = output_fc.to("cuda:0") + output_dn * (output_fc_max - output_fc_min).unsqueeze(1).to("cuda:0")
            for i in ch_idx:
                output_dn[i] = output_fc[i]
            return output_dn

            # output = self.dn(x.to('cuda:0'), z.to('cuda:0'), mode, 1, epo)
            # return output


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-F.one_hot(target, x.size()[-1]) * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

