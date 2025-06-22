'''
@File    :   dn_torch_v6_8.py
@Time    :   2023/05/24 19:00
@Author  :   Hong Shijie
@Version :   6.8
@Note    :   在v6.7的基础上，神经元可以自适应增长
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
        self.synapse_age = 20.0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.x2y = nn.Linear(self.x_neuron_num, self.y_neuron_num, bias=False)
        self.z2y = nn.Linear(self.z_neuron_num, self.y_neuron_num, bias=False)
        self.y2z = nn.Linear(self.y_neuron_num, self.z_neuron_num, bias=False)

        self.y_neuron_age = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)

        self.y_threshold = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)

        self.z_neuron_age = nn.Parameter(torch.zeros((1, self.z_neuron_num)), requires_grad=False)

        self.trans_y = Trans_y_response.apply

        self.x2y.weight.data = torch.rand(self.x2y.weight.data.shape)
        self.z2y.weight.data = torch.rand(self.z2y.weight.data.shape)
        # self.y2z.weight.data = torch.rand(self.y2z.weight.data.shape)
        self.y2z.weight.data = torch.zeros(self.y2z.weight.data.shape)

        self.z2y.weight.requires_grad = False

        self.max_index = torch.empty(0)

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
                    self.hebbian_learning(x, z_hot, max_index, max_response, batch, per_item, num)

            self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
            self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)
            x2y = self.x2y(x)
            y_activated = torch.where(self.y_neuron_age >= 1, 1, 0)
            y_activated = y_activated.to(self.device)
            y_temp = x2y.mul(y_activated)
            y_temp_cp = y_temp.data
            y = self.top_k_competition(y_temp_cp, 0)
            y_tr = self.trans_y(y_temp, y)

            output = self.y2z(y_tr)
            y_activated_num = torch.sum(y_activated)

            return output, y_activated_num

        elif mode == 'test':
            self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
            self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)

            y_bottom_up_response = self.x2y(x)
            y_activated_num = torch.where(self.y_neuron_age >= 1, 1, 0)
            y_activated_num = y_activated_num.to(self.device)
            y_pre_response = y_bottom_up_response.mul(y_activated_num)
            y_response = self.top_k_competition(y_pre_response, 0)
            output = self.y2z(y_response)
            # output = torch.argmax(output[0])
            return output

    def top_k_competition(self, y_pre_response, flag):
        if flag:
            max_response, max_index = torch.max(y_pre_response, 1)

            "存储最大激活值的索引"
            temp_mask = ~torch.isin(max_index.data.cpu(), self.max_index)
            temp_indices, _ = torch.sort(torch.unique(max_index.data.cpu()[temp_mask]))
            self.max_index = torch.cat((self.max_index, temp_indices), dim=0)

            'test'
            # self.y_neuron_age[0, [max_index[5], max_index[9]]] = 1.0

            judge = torch.where(max_response > self.y_threshold[0, max_index.data], 1, 0) \
                | torch.where(self.y_neuron_age[0, max_index.data] < self.synapse_age, 1, 0)

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

    def hebbian_learning(self, x, z_hot, max_index, max_response, batch, per_item, num):
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

        if num == per_item - 1:
            # inhibit  Y response
            y_cur_response = y_response.clone()
            y_cur_response.index_put_(
                indices=tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0)),
                values=max_response)

            y_cur_response = ((torch.sum(y_cur_response, dim=0)).div(divisor)).unsqueeze(0)

            # if torch.max(self.y_neuron_age) >= torch.tensor(19.0):
            #     print(torch.max(self.y_neuron_age, 1))

            y_lr = (torch.sum(y_lr.squeeze(2), dim=0).div(divisor)).unsqueeze(0)
            y_lr = torch.where((y_lr <= (1 / self.synapse_age)) & (y_lr > 0), 1 / (self.y_neuron_age - self.synapse_age + 2.0), 0)

            self.y_threshold.data = y_lr.mul(y_cur_response) + (1-y_lr).mul(self.y_threshold.data)

            new_age = torch.zeros(self.y_neuron_age.size()[1])
            new_age = new_age.to(self.device)

            new_age.index_put_(indices=tuple(idxs.unsqueeze(0)), values=torch.ones(cnts.size()).to(self.device))
            self.y_neuron_age.data = self.y_neuron_age.data + new_age.t()

            # y to z
            # z_lr = (1 / (self.z_neuron_age + 1.0))
            z_lr = torch.where(self.z_neuron_age > 10, 0.1, 1 / (self.z_neuron_age + 1.0))
            z_lr = z_lr.to(self.device)
            z_lr = z_hot.mul(z_lr).unsqueeze(2)

            cnts = torch.sum(z_hot, dim=0)
            divisor = torch.where(cnts > 0, cnts, 1.0)

            temp_weight = ((1 - z_lr.repeat(1, 1, self.y2z.weight.data.shape[1])).mul(self.y2z.weight.data.unsqueeze(0)) +
                           z_lr.repeat(1, 1, self.y2z.weight.data.shape[1]).mul(y_response.unsqueeze(1))) - \
                self.y2z.weight.data.unsqueeze(0).repeat(batch, 1, 1)
            temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))

            # temp_weight.data[:, torch.nonzero(self.y_neuron_age.data.squeeze(0) == 0).squeeze(1).cpu().numpy()] = 0

            self.y2z.weight.data = self.y2z.weight.data + temp_weight

            cnts = torch.where(cnts > 0, 1.0, 0)
            self.z_neuron_age.data = self.z_neuron_age.data + cnts.t()

    def reset(self, new_num):
        self.y_neuron_num = new_num

        temp_data = self.x2y.weight.data
        self.x2y = nn.Linear(self.x_neuron_num, self.y_neuron_num, bias=False)
        self.x2y.weight.data = torch.rand(self.x2y.weight.data.shape)
        # self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
        # self.x2y.weight.data[temp_data.size()[0]:-1, :] = temp_data
        # self.x2y.weight.data[temp_data.size()[0]:-1, :] = temp_data
        self.x2y.weight[:temp_data.size()[0]].data.copy_(temp_data)

        temp_data = self.z2y.weight.data
        self.z2y = nn.Linear(self.z_neuron_num, self.y_neuron_num, bias=False)
        self.z2y.weight.data = torch.rand(self.z2y.weight.data.shape)
        # self.z2y.weight.data = F.normalize(self.z2y.weight.data, dim=1)
        self.z2y.weight[:temp_data.size()[0]].data.copy_(temp_data)

        temp_data = self.y2z.weight.data
        self.y2z = nn.Linear(self.y_neuron_num, self.z_neuron_num, bias=False)
        self.y2z.weight.data = torch.zeros(self.y2z.weight.data.shape)
        # self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)
        # self.y2z.weight.data[:, temp_data.size()[1]-1:-1] = temp_data
        self.y2z.weight[:temp_data.size()[0], :temp_data.size()[1]].data.copy_(temp_data)

        temp_data = self.y_neuron_age.data
        self.y_neuron_age = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)
        # self.y_neuron_age.data[0, temp_data.size()[1]-1:-1] = temp_data
        self.y_neuron_age[:temp_data.size()[0], :temp_data.size()[1]].data.copy_(temp_data)

        temp_data = self.y_threshold.data
        self.y_threshold = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)
        # self.y_threshold.data[0, temp_data.size()[1]-1:-1] = temp_data
        self.y_threshold[:temp_data.size()[0], :temp_data.size()[1]].data.copy_(temp_data)

