'''
@File    :   backbone_v3.py
@Time    :   2023/07/22 15:30
@Author  :   Hong Shijie
@Version :   3
@Note    :   resnet18自搭建
'''
# from torch import nn
# import torchvision
# import torch
# import torch.nn.functional as F
#
# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1) -> None:
#         super(BasicBlock, self).__init__()
#         # 残差部分
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
#
#         # shortcut 部分
#         # 由于存在维度不一致的情况 所以分情况
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 # 卷积核为1 进行升降维
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
# #         print('shape of x: {}'.format(x.shape))
#         out = self.layer(x)
# #         print('shape of out: {}'.format(out.shape))
# #         print('After shortcut shape of x: {}'.format(self.shortcut(x).shape))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
# class ResNet18(nn.Module):
#     def __init__(self, num_classes=10) -> None:
#         super(ResNet18, self).__init__()
#         self.in_channels = 64
#         # 第一层作为单独的 因为没有残差快
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         # conv2_x
#         self.conv2 = self._make_layer(BasicBlock, 64, [[1, 1], [1, 1]])
#         # self.conv2_2 = self._make_layer(BasicBlock,64,[1,1])
#
#         # conv3_x
#         self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1],[1, 1]])
#         # self.conv3_2 = self._make_layer(BasicBlock,128,[1,1])
#
#         # conv4_x
#         self.conv4 = self._make_layer(BasicBlock, 256, [[2, 1], [1, 1]])
#         # self.conv4_2 = self._make_layer(BasicBlock,256,[1,1])
#
#         # conv5_x
#         self.conv5 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]])
#         # self.conv5_2 = self._make_layer(BasicBlock,512,[1,1])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)
#
#     #这个函数主要是用来，重复同一个残差块
#     def _make_layer(self, block, out_channels, strides):
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.conv5(out)
#
# #         out = F.avg_pool2d(out,7)
#         out = self.avgpool(out)
#         out = out.reshape(x.shape[0], -1)
#         out = self.fc(out)
#         return out


import torch
from torch import nn

# 基础块
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear
import torchvision


class BasicBlock(nn.Module):

    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        stride = 1
        _features = out_features
        if self.in_features != self.out_features:
            # 在输入通道和输出通道不相等的情况下计算通道是否为2倍差值
            if self.out_features / self.in_features == 2.0:
                stride = 2  # 在输出特征是输入特征的2倍的情况下 要想参数不翻倍 步长就必须翻倍
            else:
                raise ValueError("输出特征数最多为输入特征数的2倍！")

        self.conv1 = Conv2d(in_features, _features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(_features, _features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # 下采样
        self.downsample = None if self.in_features == self.out_features else nn.Sequential(
            Conv2d(in_features, out_features, kernel_size=1, stride=2, bias=False),
            BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 输入输出的特征数不同时使用下采样层
        if self.in_features != self.out_features:
            identity = self.downsample(x)

        # 残差求和
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(1, 1, 0)
        # self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # <---- 输出为{Tensor:(64,512,1,1)}
        x = torch.flatten(x, 1)  # <----------------这里是个坑 很容易漏 从池化层到全连接需要一个压平 输出为{Tensor:(64,512)}
        x = self.fc(x)  # <------------ 输出为{Tensor:(64,10)}
        return x

    def load_pretrained_weights(self):
        model = torchvision.models.resnet18(pretrained=True)
        weights = model.state_dict()
        weights.pop('conv1.weight')
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        return weights
#
#
# # 模型数据验证
#
# if __name__ == "__main__":
#     mode = ResNet18()
#     print(mode)
#     data = torch.ones((64, 3, 32, 32))
#     output = mode(data)
#     print(output.shape)

# import torch
# import torch.nn as nn
# from torch import Tensor
# from typing import Type, Any, Callable, Union, List, Optional
#
#
# def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
#     """3x3 convolution with padding"""
#     return nn.Conv2d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=dilation,
#         groups=groups,
#         bias=False,
#         dilation=dilation,
#     )
#
#
# class BasicBlock(nn.Module):
#     expansion: int = 1
#
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#     def __init__(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         layers: List[int],
#         num_classes: int = 1000,
#         zero_init_residual: bool = False,
#         groups: int = 1,
#         width_per_group: int = 64,
#         replace_stride_with_dilation: Optional[List[bool]] = None,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError(
#                 "replace_stride_with_dilation should be None "
#                 f"or a 3-element tuple, got {replace_stride_with_dilation}"
#             )
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck) and m.bn3.weight is not None:
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
#
#     def _make_layer(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         planes: int,
#         blocks: int,
#         stride: int = 1,
#         dilate: bool = False,
#     ) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer,
#                 )
#             )
#
#         return nn.Sequential(*layers)
#
#     def _forward_impl(self, x: Tensor) -> Tensor:
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)