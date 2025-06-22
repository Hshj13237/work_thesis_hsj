'''
@File    :   dn_torch_v4.py
@Time    :   2023/04/04 10:00
@Author  :   Hong Shijie
@Version :   4.0
@Note    :   在v2的基础上，更改了forward的结构使其适应于与cnn与dn的结合，并增加了额外的类用于适应反向传播，同时train时设置成不进行反向传播
'''

import torch
from torch import nn
import torch.nn.functional as F


class Trans_y_response(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pre, y_act):
        ctx.save_for_backward(y_act)
        return y_act

    @staticmethod
    def backward(ctx, grad_output):
        y_act, = ctx.saved_tensors
        grad_y_pre = grad_output.mul(y_act)
        grad_y_act = None
        return grad_y_pre, grad_y_act


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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.x2y = nn.Linear(self.x_neuron_num, self.y_neuron_num, bias=False)
        self.z2y = nn.Linear(self.z_neuron_num, self.y_neuron_num, bias=False)
        self.y2z = nn.Linear(self.y_neuron_num, self.z_neuron_num, bias=False)

        # self.y_neuron_age = torch.zeros((1, self.y_neuron_num))
        self.y_neuron_age = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)

        # self.y_threshold = torch.zeros((1, self.y_neuron_num))
        self.y_threshold = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)
        # self.y_threshold = self.y_threshold.to(self.device)

        # self.z_neuron_age = torch.zeros((1, self.z_neuron_num))
        self.z_neuron_age = nn.Parameter(torch.zeros((1, self.z_neuron_num)), requires_grad=False)

        self.trans_y = Trans_y_response.apply

        self.x2y.weight.data = torch.rand(self.x2y.weight.data.shape)
        self.z2y.weight.data = torch.rand(self.z2y.weight.data.shape)
        self.y2z.weight.data = torch.rand(self.y2z.weight.data.shape)

        self.z2y.weight.requires_grad = False

    def forward(self, x, z, mode, per_item):
        batch = int(x.shape[0])
        x = x.view(batch, -1)
        x = F.normalize(x, dim=1)
        z_hot = F.one_hot(z, num_classes=self.z_neuron_num)
        z_hot = F.normalize(z_hot.float(), dim=1)

        if mode == 'train':
            with torch.no_grad():
                for num in range(per_item):
                    self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
                    self.z2y.weight.data = F.normalize(self.z2y.weight.data, dim=1)
                    self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)

                    y_bottom_up_response = self.x2y(x)
                    y_top_down_response = self.z2y(z_hot)
                    y_pre_response = self.y_bottom_up_percent * y_bottom_up_response + self.y_top_down_percent * y_top_down_response
                    max_response, max_index = self.top_k_competition(y_pre_response, 1)
                    # print(max_response, max_index)
                    self.hebbian_learning(x, z_hot, max_index, max_response)

            self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
            self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)
            x2y = self.x2y(x)
            y_activated = torch.where(self.y_neuron_age >= 1, 1, 0)
            y_activated = y_activated.to(self.device)
            y_temp = x2y * y_activated
            y_temp_cp = y_temp.data

            y = self.top_k_competition(y_temp_cp, 0)
            # print(torch.max(y, 1))
            y_tr = self.trans_y(y_temp, y)

            output = self.y2z(y_tr)
            y_activated_num = torch.sum(y_activated)
            # print(y_activated_num)
            # print('------------------------')
            return output, y_activated_num

        elif mode == 'test':
            self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
            self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)

            y_bottom_up_response = self.x2y(x)
            y_activated_num = torch.where(self.y_neuron_age >= 1, 1, 0)
            y_activated_num = y_activated_num.to(self.device)
            y_pre_response = y_bottom_up_response * y_activated_num
            y_response = self.top_k_competition(y_pre_response, 0)
            output = self.y2z(y_response)
            output = torch.argmax(output[0])
            return output

        # for num in range(per_item):
        #     batch = int(x.shape[0])
        #     x = x.view(batch, -1)
        #     x = F.normalize(x, dim=1)
        #     z_hot = F.one_hot(z, num_classes=self.z_neuron_num)
        #     z_hot = F.normalize(z_hot.float(), dim=1)
        #
        #     self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
        #     self.z2y.weight.data = F.normalize(self.z2y.weight.data, dim=1)
        #     self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)
        #
        #     y_bottom_up_response = self.x2y(x)
        #
        #     if mode == 'train':
        #         y_top_down_response = self.z2y(z_hot)
        #
        #         y_pre_response = self.y_bottom_up_percent * y_bottom_up_response + self.y_top_down_percent * y_top_down_response
        #
        #         max_response, max_index = self.top_k_competition(y_pre_response, 1)
        #         self.hebbian_learning(x, z_hot, max_index, max_response)
        #
        #         y_activated_num = torch.where(self.y_neuron_age >= 1, 1, 0)
        #         y_activated_num = torch.sum(y_activated_num)
        #
        #         return y_activated_num
        #     elif mode == 'test':
        #         y_activated_num = torch.where(self.y_neuron_age >= 1, 1, 0)
        #         y_activated_num = y_activated_num.to(self.device)
        #         y_pre_response = y_bottom_up_response * y_activated_num
        #
        #         y_pre_response_cp = y_pre_response.data
        #
        #         y_response = self.top_k_competition(y_pre_response_cp, 0)
        #
        #         y_response_tr = self.trans_y(y_pre_response, y_response)
        #
        #         output = self.y2z(y_response_tr)
        #         output = torch.argmax(output[0])
        #         return output

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

    def extra_repr(self):
        return f'(y_neuron_age): {self.y_neuron_age.size()} \n(z_neuron_age): {self.z_neuron_age.size()}\n' \
               f'(y_threshold): {self.y_threshold.size()}'

    def __getstate__(self):
        return {
            'x2y_weight': self.x2y.weight, 'z2y_weight': self.z2y.weight, 'y2z_weight': self.y2z.weight,
            'y_neuron_age': self.y_neuron_age, 'z_neuron_age': self.z_neuron_age, 'y_threshold': self.y_threshold
        }

    def __setstate__(self, state):
        self.x2y.weight = state['x2y_weight']
        self.z2y.weight = state['z2y_weight']
        self.y2z.weight = state['y2z_weight']
        self.y_neuron_age = state['y_neuron_age']
        self.z_neuron_age = state['z_neuron_age']
        self.y_threshold = state['y_threshold']


    # def backward_hook_func(self, module, grad_input, grad_output):
    #     print(module)
    #     # for y in grad_output:
    #     #     print('output')
    #     #     print(y.shape)
    #     #     print(y)
    #     # for x in grad_input:
    #     #     print('input')
    #     #     print(x.shape)
    #     #     print(x)




