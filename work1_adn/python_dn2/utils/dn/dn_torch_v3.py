'''
@File    :   train_mnist.py
@Time    :   2023/03/19 23:00
@Author  :   Hong Shijie
@Version :   3.0
@Note    :   基于dn_torch_v2.py,增加突触维护
'''

import torch
from torch import nn
import numpy as np

import torch.nn.functional as F

class DN(nn.Module):
    def __init__(self, input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag):
        super(DN, self).__init__()
        self.x_neuron_num = input_dim[0] * input_dim[1]
        self.y_neuron_num = y_neuron_num
        self.z_neuron_num = z_neuron_num
        self.y_top_k = y_top_k
        self.y_bottom_up_percent = y_bottom_up_percent
        self.y_top_down_percent = 1.0 - self.y_bottom_up_percent
        self.synapse_age = 20
        self.synapse_coefficient = [0.8, 1.2]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.x2y = nn.Linear(self.x_neuron_num, self.y_neuron_num, bias=False)
        self.z2y = nn.Linear(self.z_neuron_num, self.y_neuron_num, bias=False)
        self.y2z = nn.Linear(self.y_neuron_num, self.z_neuron_num, bias=False)

        self.x2y_diff = torch.zeros(self.x2y.weight.data.shape)
        self.x2y_diff = self.x2y_diff.to(self.device)
        self.x2y_factor = torch.ones(self.x2y.weight.data.shape)
        self.x2y_factor = self.x2y_factor.to(self.device)

        self.y_neuron_age = torch.zeros((1, self.y_neuron_num))

        self.y_threshold = torch.zeros((1, self.y_neuron_num))
        self.y_threshold = self.y_threshold.to(self.device)

        self.z_neuron_age = torch.zeros((1, self.z_neuron_num))


        # self.x2y_norm = nn.LayerNorm([self.x2y.weight.shape[0], self.x2y.weight.shape[1]], elementwise_affine=False)
        # self.z2y_norm = nn.LayerNorm([self.z2y.weight.shape[0], self.z2y.weight.shape[1]], elementwise_affine=False)
        # self.y2z_norm = nn.LayerNorm([self.y2z.weight.shape[0], self.y2z.weight.shape[1]], elementwise_affine=False)


        # nn.init.normal_(self.x2y.weight, mean=0.1307, std=0.3081)
        # nn.init.normal_(self.z2y.weight, mean=0.1307, std=0.3081)
        # nn.init.normal_(self.y2z.weight, mean=0.1307, std=0.3081)

        self.x2y.weight.data = torch.rand(self.x2y.weight.data.shape)
        self.z2y.weight.data = torch.rand(self.z2y.weight.data.shape)
        self.y2z.weight.data = torch.rand(self.y2z.weight.data.shape)

    def forward(self, x, z, mode, per_item):

        for num in range(per_item):
            batch = int(x.shape[0])
            x = x.view(batch, -1)
            x = F.normalize(x, dim=1)
            z_hot = F.one_hot(z, num_classes=self.z_neuron_num)
            z_hot = F.normalize(z_hot.float(), dim=1)

            temp_x2y_weight = self.x2y.weight.data
            # 改动 3.19-21：30
            # self.x2y.weight.data = self.x2y.weight.data.mul(self.x2y_factor.mul(self.x2y_factor))
            self.x2y.weight.data = self.x2y.weight.data.mul(self.x2y_factor)

            self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
            self.z2y.weight.data = F.normalize(self.z2y.weight.data, dim=1)
            self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)

            y_bottom_up_response = self.x2y(x)
            self.x2y.weight.data = temp_x2y_weight

            if mode == 'train':
                y_top_down_response = self.z2y(z_hot)

                y_pre_response = self.y_bottom_up_percent * y_bottom_up_response + self.y_top_down_percent * y_top_down_response

                max_response, max_index = self.top_k_competition(y_pre_response, 1)
                self.hebbian_learning(x, z_hot, max_index, max_response)

                y_activated_num = torch.where(self.y_neuron_age >= 1, 1, 0)
                y_activated_num = torch.sum(y_activated_num)

                return y_activated_num
            elif mode == 'test':
                y_activated_num = torch.where(self.y_neuron_age > 0, 1, 0)
                y_activated_num = y_activated_num.to(self.device)
                y_pre_response = y_bottom_up_response * y_activated_num
                y_response = self.top_k_competition(y_pre_response, 0)
                output = self.y2z(y_response)
                output = torch.argmax(output[0])
                return output


    def top_k_competition(self, y_pre_response, flag):
        # y_response = torch.zeros(y_pre_response.shape)
        if flag:
            max_response, max_index = torch.max(y_pre_response, 1)
            if max_response > self.y_threshold[0][max_index.item()]:
                # y_response[0][max_index.item()] = 1.0
                return max_response, max_index
            elif self.y_neuron_age[0][max_index.item()] < 1.0:
                # y_response[0][max_index.item()] = 1.0
                return max_response, max_index
            else:
                unactivated_flag = torch.where(self.y_neuron_age >= 1, 0., 1.)
                unactivated_flag = unactivated_flag.to(self.device)
                if torch.sum(unactivated_flag, dim=1) > 0:
                    y_pre_response = torch.mul(y_pre_response, unactivated_flag)
                    max_index = torch.argmax(y_pre_response, dim=1)
                    # y_response[0][max_index.item()] = 1.0
                    return max_response, max_index
                else:
                    # y_response[0][max_index.item()] = 1.0
                    return max_response, max_index
        else:
            y_response = torch.zeros(y_pre_response.shape)
            y_response = y_response.to(self.device)

            max_response, max_index = torch.max(y_pre_response, 1)
            y_response[0][max_index] = 1.0

            return y_response

    def hebbian_learning(self, x, z_hot, max_index, max_response):
        y_response = (torch.zeros(self.y_neuron_age.shape)).t()
        y_response = y_response.to(self.device)
        y_response[max_index.item()][0] = 1.0

        y_lr_temp = (1 / (self.y_neuron_age + 1.0)).t()
        y_lr_temp = y_lr_temp.to(self.device)

        # x to y
        y_lr = (y_lr_temp * y_response).repeat(1, self.x2y.weight.data.shape[1])
        self.x2y.weight.data = (1 - y_lr).mul(self.x2y.weight.data) + y_lr.mul(x)
        self.x2y_diff = \
            (1 - y_lr).mul(self.x2y_diff) + \
            y_lr.mul(torch.abs(self.x2y.weight.data - self.x2y_diff))
        synapse_flag = self.y_neuron_age.t().repeat(1, self.x2y_diff.shape[1])
        synapse_flag = synapse_flag.to(self.device)


        new_synapse_factor = self.synapse_maintainence(self.x2y_diff)
        self.x2y_factor = torch.where(synapse_flag > self.synapse_age, new_synapse_factor, self.x2y_factor)

        # z to y
        y_lr = (y_lr_temp * y_response).repeat(1, self.z2y.weight.data.shape[1])
        self.z2y.weight.data = (1 - y_lr).mul(self.z2y.weight.data)+ y_lr.mul(z_hot)

        self.y_neuron_age[0][max_index.item()] += 1.0

        # y to z
        z_lr_temp = (1 / (self.z_neuron_age + 1.0)).t()
        z_lr_temp = z_lr_temp.to(self.device)
        z_lr = (z_lr_temp * z_hot.t()).repeat(1, self.y2z.weight.data.shape[1])
        self.y2z.weight.data = (1 - z_lr).mul(self.y2z.weight.data) + z_lr.mul(y_response.t())
        self.z_neuron_age[0][(torch.argmax(z_hot)).item()] += 1.0

        # inhibit  Y response
        self.y_threshold[0][max_index.item()] = y_lr_temp[max_index.item()][0] * max_response +\
            (1 - y_lr_temp[max_index.item()][0]) * self.y_threshold[0][max_index.item()]

    def synapse_maintainence(self, synapse_diff):
        mean_diff = torch.mean(synapse_diff, dim=1)

        lower_thresh = (self.synapse_coefficient[0] * mean_diff).unsqueeze(0).t()
        lower_thresh = lower_thresh.repeat(1, synapse_diff.shape[1])
        upper_thresh = (self.synapse_coefficient[1] * mean_diff).unsqueeze(0).t()
        upper_thresh = upper_thresh.repeat(1, synapse_diff.shape[1])

        upper_flag = torch.where(synapse_diff > upper_thresh, 0., 1.)
        middle_factor = (synapse_diff - upper_thresh) / torch.maximum((lower_thresh - upper_thresh), torch.tensor(1e-12))
        new_synapse_factor = torch.where(synapse_diff < lower_thresh, 1., middle_factor * upper_flag)

        return new_synapse_factor







