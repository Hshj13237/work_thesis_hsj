'''
@File    :   backbone_v1_1.py
@Time    :   2023/06/06 15:00
@Author  :   Hong Shijie
@Version :   1.1
@Note    :   在v1的基础上，在resnet后加入了grl层
'''
from torch import nn
import torchvision
import torch
from utils.modules.grl import GRL
import torch.nn.functional as F


class backbone(nn.Module):
    def __init__(self, output_channel, batch_size):
        super(backbone, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(self.resnet18.fc.in_features, output_channel)
        self.resnet18_revise()

        self.grl = GRL(lambda_=0.5)

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

    def resnet18_revise(self):
        self.resnet18.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)

        self.resnet18.maxpool = nn.MaxPool2d(1, 1, 0)

        self.resnet18.fc = nn.Sequential()

    def forward(self, x):
        x = self.resnet18(x)
        x = self.grl(x)
        x = self.fc(x)
        return x


