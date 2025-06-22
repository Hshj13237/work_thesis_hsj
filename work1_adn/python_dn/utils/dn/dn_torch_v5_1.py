'''
@File    :   dn_torch_v5_1.py
@Time    :   2023/04/23 19:00
@Author  :   Hong Shijie
@Version :   5.1
@Note    :   在v5的基础上，将自适应阈值模块中的同一神经元多激活值的处理方法，由均值转换为最小值
'''

import torch
from torch import nn
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.x2y = nn.Linear(self.x_neuron_num, self.y_neuron_num, bias=False)
        self.z2y = nn.Linear(self.z_neuron_num, self.y_neuron_num, bias=False)
        self.y2z = nn.Linear(self.y_neuron_num, self.z_neuron_num, bias=False)

        self.y_neuron_age = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)

        self.y_threshold = torch.zeros((1, self.y_neuron_num))
        self.y_threshold = self.y_threshold.to(self.device)

        self.z_neuron_age = nn.Parameter(torch.zeros((1, self.z_neuron_num)), requires_grad=False)


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

            self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
            self.z2y.weight.data = F.normalize(self.z2y.weight.data, dim=1)
            self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)

            y_bottom_up_response = self.x2y(x)

            if mode == 'train':
                y_top_down_response = self.z2y(z_hot)

                y_pre_response = self.y_bottom_up_percent * y_bottom_up_response + self.y_top_down_percent * y_top_down_response

                max_response, max_index = self.top_k_competition(y_pre_response, 1)
                self.hebbian_learning(x, z_hot, max_index, max_response, batch)

                y_activated_num = torch.where(self.y_neuron_age >= 1, 1, 0)
                y_activated_num = torch.sum(y_activated_num)

                return y_activated_num, max_response
            elif mode == 'test':
                y_activated_num = torch.where(self.y_neuron_age >= 1, 1, 0)
                y_activated_num = y_activated_num.to(self.device)
                # y_pre_response = y_bottom_up_response * y_activated_num
                y_pre_response = y_bottom_up_response
                y_response = self.top_k_competition(y_pre_response, 0)
                output = self.y2z(y_response)
                # output = torch.argmax(output, dim=1)
                return output

    def top_k_competition(self, y_pre_response, flag):
        if flag:
            max_response, max_index = torch.max(y_pre_response, 1)

            'test'
            # self.y_neuron_age[0, [max_index[5], max_index[9]]] = 1.0

            judge = torch.where(max_response > self.y_threshold[0, max_index.data], 1, 0) \
                | torch.where(self.y_neuron_age[0, max_index.data] < 1.0, 1, 0)

            add_index = torch.where(judge < 1)
            # add_index = torch.stack(add_index)

            if torch.sum(judge) < len(judge):
                unactivated_flag = torch.where(self.y_neuron_age >= 1, 0., 1.).repeat(len(add_index), 1)
                unactivated_flag = unactivated_flag.to(self.device)

                if torch.sum(unactivated_flag, dim=1) > 0:
                    y_temp_response = y_pre_response[torch.stack(add_index), :].squeeze(0)
                    y_temp_response = torch.mul(y_temp_response, unactivated_flag)
                    max_index.index_put_(indices=add_index, values=torch.argmax(y_temp_response, dim=1))
                    # max_index.index_copy_(0, add_index.squeeze(0), torch.argmax(y_temp_response, dim=1))

            return max_response, max_index

        else:
            y_response = torch.zeros(y_pre_response.shape)
            y_response = y_response.to(self.device)

            max_response, max_index = torch.max(y_pre_response, 1)

            index = tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0))
            y_response.index_put_(indices=index, values=(torch.ones(max_index.size())).to(self.device))

            return y_response

    def hebbian_learning(self, x, z_hot, max_index, max_response, batch):
        y_response = (torch.zeros(self.y_neuron_age.size())).repeat(batch, 1)
        y_response = y_response.to(self.device)
        # y_response = (torch.zeros(self.y_neuron_age.size())).t()
        # y_response = y_response.to(self.device)
        # index = tuple(torch.stack([torch.arange(0, max_index.size()[0]), max_index], dim=0))
        # idxs = tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0))
        y_response.index_put_(
            indices=tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0)),
            values=(torch.ones(max_index.size())).to(self.device))

        "test"
        # self.y_neuron_age[0, [max_index[5], max_index[9]]] = 1.0

        y_lr = (1 / (self.y_neuron_age + 1.0))
        y_lr = y_lr.to(self.device)
        y_lr = y_response.mul(y_lr).unsqueeze(2)

        idxs, cnts = torch.unique(max_index, return_counts=True)
        divisor = torch.ones(self.y_neuron_age.size()[1])
        divisor = divisor.to(self.device)
        divisor.index_put_(indices=tuple(idxs.unsqueeze(0)), values=cnts.to(torch.float32))

        # x to y
        temp_weight = ((1 - y_lr.repeat(1, 1, self.x2y.weight.data.shape[1])).mul(self.x2y.weight.data.unsqueeze(0)) +
                       y_lr.repeat(1, 1, self.x2y.weight.data.shape[1]).mul(x.unsqueeze(1))) - \
            self.x2y.weight.data.unsqueeze(0).repeat(batch, 1, 1)
        temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))

        self.x2y.weight.data = self.x2y.weight.data + temp_weight

        # z to y
        temp_weight = ((1 - y_lr.repeat(1, 1, self.z2y.weight.data.shape[1])).mul(self.z2y.weight.data.unsqueeze(0)) +
                       y_lr.repeat(1, 1, self.z2y.weight.data.shape[1]).mul(z_hot.unsqueeze(1))) - \
            self.z2y.weight.data.unsqueeze(0).repeat(batch, 1, 1)
        temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))

        self.z2y.weight.data = self.z2y.weight.data + temp_weight

        new_age = torch.zeros(self.y_neuron_age.size()[1])
        new_age = new_age.to(self.device)
        new_age.index_put_(indices=tuple(idxs.unsqueeze(0)), values=cnts.to(torch.float32))
        self.y_neuron_age.data = self.y_neuron_age.data + new_age.t()

        # inhibit  Y response
        y_cur_response = (torch.ones(y_response.size()) * 3).to(self.device)
        y_cur_response.index_put_(
            indices=tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0)),
            values=max_response)

        "mean"
        # y_cur_response = ((torch.sum(y_cur_response, dim=0)).div(divisor)).unsqueeze(0)

        "min"
        y_cur_response, _ = torch.min(y_cur_response, dim=0)
        y_cur_response = torch.where(y_cur_response > 2, 0, y_cur_response)
        y_cur_response = y_cur_response.unsqueeze(0)

        y_lr = (torch.sum(y_lr.squeeze(2), dim=0).div(divisor)).unsqueeze(0)

        self.y_threshold = y_lr.mul(y_cur_response) + (1-y_lr).mul(self.y_threshold)

        # self.y_threshold[0][max_index.item()] = y_lr_temp[max_index.item()][0] * max_response +\
        #     (1 - y_lr_temp[max_index.item()][0]) * self.y_threshold[0][max_index.item()]


        # y to z
        z_lr = (1 / (self.z_neuron_age + 1.0))
        z_lr = z_lr.to(self.device)
        z_lr = z_hot.mul(z_lr).unsqueeze(2)

        cnts = torch.sum(z_hot, dim=0)
        divisor = torch.where(cnts > 0, cnts, 1.0)

        temp_weight = ((1 - z_lr.repeat(1, 1, self.y2z.weight.data.shape[1])).mul(self.y2z.weight.data.unsqueeze(0)) +
                       z_lr.repeat(1, 1, self.y2z.weight.data.shape[1]).mul(y_response.unsqueeze(1))) - \
            self.y2z.weight.data.unsqueeze(0).repeat(batch, 1, 1)
        temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))

        self.y2z.weight.data = self.y2z.weight.data + temp_weight
        self.z_neuron_age.data = self.z_neuron_age.data + cnts.t()








