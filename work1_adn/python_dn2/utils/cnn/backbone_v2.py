'''
@File    :   backbone_v2.py
@Time    :   2023/05/09 15:30
@Author  :   Hong Shijie
@Version :   2.0
@Note    :   在v1的基础上，将末尾全连接层变为两层
'''
from torch import nn
import torchvision
import torch
import torch.nn.functional as F

class backbone(nn.Module):
    def __init__(self, output_channel1, output_channel2, batch_size):
        super(backbone, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18_revise(output_channel1)
        self.fc = nn.Linear(output_channel1, output_channel2)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if batch_size == 1:
            self.batchnorm_2_instancenorm()

    def batchnorm_2_instancenorm(self):
        bn_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_dict[name] = module.num_features

        for key in bn_dict.keys():
            values = key.split('.')
            name = 'self'
            for v in values:
                if v.isdecimal():
                    name += '[{}]'.format(v)
                else:
                    name += '.{}'.format(v)
            exec(name + '=' + 'nn.InstanceNorm2d({}, affine=True, track_running_stats=True)'.format(bn_dict[key]))

    def resnet18_revise(self, output_channel1):
        self.resnet18.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)

        self.resnet18.maxpool = nn.MaxPool2d(1, 1, 0)

        nun_fc_in_ch = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(nun_fc_in_ch, output_channel1)


    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)
        return x


