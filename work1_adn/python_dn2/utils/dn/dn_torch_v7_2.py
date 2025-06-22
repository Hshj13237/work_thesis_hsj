'''
@File    :   dn_torch_v7_2.py
@Time    :   2023/06/27 19:00
@Author  :   Hong Shijie
@Version :   7.2
@Note    :   在v6.11和v7的基础上，加入注意力机制，并同时计算top-k1\2\3
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
        self.synapse_age = 1.0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.x2y = nn.Linear(self.x_neuron_num, self.y_neuron_num, bias=False)
        self.z2y = nn.Linear(self.z_neuron_num, self.y_neuron_num, bias=False)
        self.y2z = nn.Linear(self.y_neuron_num, self.z_neuron_num, bias=False)

        self.y_neuron_age = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)

        self.y_threshold = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)

        self.z_neuron_age = nn.Parameter(torch.zeros((1, self.z_neuron_num)), requires_grad=False)

        self.x2y.weight.data = torch.rand(self.x2y.weight.data.shape)
        self.z2y.weight.data = torch.rand(self.z2y.weight.data.shape)
        # self.y2z.weight.data = torch.rand(self.y2z.weight.data.shape)
        self.y2z.weight.data = torch.zeros(self.y2z.weight.data.shape)

        self.z2y.weight.requires_grad = False

        self.max_index = torch.empty(0)

        self.z_lr_limit = 0.2

    def forward(self, x, z, mode, per_item, global_step, test_cnt):
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
                    # self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=0)

                    if num == 0:
                        inputs = x
                    else:
                        inputs = x.mul(att_weight)

                    # inputs = x

                    y_bottom_up_response = self.x2y(inputs)
                    y_top_down_response = self.z2y(z_hot)

                    if num == 0:
                        y_pre_response = self.y_bottom_up_percent * y_bottom_up_response + self.y_top_down_percent * y_top_down_response
                    else:
                        y_pre_response = y_bottom_up_response

                    for cnt in range(2):
                        max_response, max_index, lower_index = self.top_k_competition(y_pre_response, 1, cnt)
                        self.hebbian_learning(inputs, z_hot, max_index, max_response, batch, per_item, num, lower_index, global_step)
                    # att_weight = self.attention(max_index)

                    y_activated = torch.where(self.y_neuron_age >= 1, 1, 0)
                    y_activated = y_activated.to(self.device)

                    att_weight = self.attention(y_pre_response, y_activated, z, mode)

                y_activated_num = torch.sum(y_activated)

                # self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
                # # self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)
                # y_bottom_up_response = self.x2y(x)
                # y_pre_response = y_bottom_up_response.mul(y_activated)
                # y_response = self.top_k_competition(y_pre_response, 0)
                # output = self.y2z(y_response)

            return att_weight, y_activated_num

        elif mode == 'test':
            with torch.no_grad():
                self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
                # self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)

                y_bottom_up_response = self.x2y(x)
                y_activated_num = torch.where(self.y_neuron_age >= 1, 1, 0)
                y_activated_num = y_activated_num.to(self.device)
                y_pre_response = y_bottom_up_response.mul(y_activated_num)
                y_response, max_index = self.top_k_competition(y_pre_response, 0, test_cnt)
                output = self.y2z(y_response)

                _, z_pre = torch.max(self.y2z.weight.t()[max_index], dim=1)
                att_weight = self.attention(y_pre_response, y_activated_num, z_pre, mode)
                # att_weight = self.attention(y_pre_response, y_activated_num, z, mode)
            return output, att_weight

    def top_k_competition(self, y_pre_response, flag, cnt):
        if flag:
            # max_response, max_index = torch.max(y_pre_response, 1)
            y_values, y_indices = torch.sort(y_pre_response, dim=1, descending=True)
            max_response, max_index = y_values[:, cnt], y_indices[:, cnt]

            "存储batch中最大激活值的索引"
            self.max_index_batch = max_index

            "存储最大激活值的索引"
            temp_mask = ~torch.isin(max_index.data.cpu(), self.max_index)
            temp_indices, _ = torch.sort(torch.unique(max_index.data.cpu()[temp_mask]))
            self.max_index = torch.cat((self.max_index, temp_indices), dim=0)

            'test'
            # self.y_neuron_age[0, [max_index[5], max_index[9]]] = 1.0

            judge = torch.where(max_response >= self.y_threshold[0, max_index.data], 1, 0) \
                | torch.where(self.y_neuron_age[0, max_index.data] < self.synapse_age, 1, 0)

            add_index = torch.where(judge < 1)
            # add_index = torch.stack(add_index)
            lower_index = torch.cat((torch.stack(add_index), max_index[add_index].unsqueeze(0)), dim=0)

            if torch.sum(judge) < len(judge):
                unactivated_flag = torch.where(self.y_neuron_age >= 1, 0., 1.).repeat(len(add_index), 1)
                unactivated_flag = unactivated_flag.to(self.device)

                if torch.sum(unactivated_flag, dim=1) > 0:
                    y_temp_response = y_pre_response[torch.stack(add_index), :].squeeze(0)
                    y_temp_response = torch.mul(y_temp_response, unactivated_flag)
                    max_index.index_put_(indices=add_index, values=torch.argmax(y_temp_response, dim=1))
                    # max_index.index_copy_(0, add_index.squeeze(0), torch.argmax(y_temp_response, dim=1))

            return max_response, max_index, lower_index

        else:
            y_response = torch.zeros(y_pre_response.shape)
            y_response = y_response.to(self.device)

            # max_response, max_index = torch.max(y_pre_response, 1)
            y_values, y_indices = torch.sort(y_pre_response, dim=1, descending=True)
            max_response, max_index = y_values[:, cnt], y_indices[:, cnt]

            index = tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0))
            y_response.index_put_(indices=index, values=(torch.ones(max_index.size())).to(self.device))

            return y_response, max_index

    def hebbian_learning(self, x, z_hot, max_index, max_response, batch, per_item, num, lower_index, global_step):
        y_response = (torch.zeros(self.y_neuron_age.size())).repeat(batch, 1)
        y_response = y_response.to(self.device)
        # y_response = (torch.zeros(self.y_neuron_age.size())).t()
        # y_response = y_response.to(self.device)
        # index = tuple(torch.stack([torch.arange(0, max_index.size()[0]), max_index], dim=0))
        # idxs = tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0))
        y_response.index_put_(
            indices=tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0)),
            values=(torch.ones(max_index.size())).to(self.device))

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

        # y to z part 1
        z_lr = 0.2 - 20 * ((self.z_neuron_age / (torch.sum(self.z_neuron_age) + 1)) - 0.01)
        z_lr = torch.where(z_lr < 0.001, 0.001, z_lr)
        z_lr = z_lr.to(self.device)
        z_lr = z_hot.mul(z_lr).unsqueeze(2).repeat(1, 1, self.y2z.weight.data.shape[1])
        z_lr = (10 * z_lr).mul(y_lr.squeeze(2).unsqueeze(1))
        z_lr = torch.where(z_lr > 0.5, 0.5, z_lr)

        if num == per_item - 1:
            # inhibit  Y response
            y_cur_response = y_response.clone()
            y_cur_response.index_put_(
                indices=tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0)),
                values=max_response)

            y_cur_response = ((torch.sum(y_cur_response, dim=0)).div(divisor)).unsqueeze(0)

            y_lr = (torch.sum(y_lr.squeeze(2), dim=0).div(divisor)).unsqueeze(0)
            y_lr = torch.where((y_lr <= (1 / self.synapse_age)) & (y_lr > 0), 1 / (self.y_neuron_age - self.synapse_age + 2.0), 0)

            self.y_threshold.data = y_lr.mul(y_cur_response) + (1-y_lr).mul(self.y_threshold.data)

            new_age = torch.zeros(self.y_neuron_age.size()[1])
            new_age = new_age.to(self.device)

            new_age.index_put_(indices=tuple(idxs.unsqueeze(0)), values=torch.ones(cnts.size()).to(self.device))
            self.y_neuron_age.data = self.y_neuron_age.data + new_age.t()

        # y to z part 2
        cnts = torch.sum(z_hot, dim=0)
        divisor = torch.where(cnts > 0, cnts, 1.0)

        temp_weight = ((1 - z_lr).mul(self.y2z.weight.data.unsqueeze(0)) + z_lr.mul(y_response.unsqueeze(1))) - \
            self.y2z.weight.data.unsqueeze(0).repeat(batch, 1, 1)
        temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))

        # temp_weight.data[:, torch.nonzero(self.y_neuron_age.data.squeeze(0) == 0).squeeze(1).cpu().numpy()] = 0

        self.y2z.weight.data = self.y2z.weight.data + temp_weight

        # cnts = torch.where(cnts > 0, 1.0, 0)
        self.z_neuron_age.data = self.z_neuron_age.data + cnts.t()

        "减弱阈值"
        if torch.numel(lower_index) != 0:
            y_response = (torch.zeros(self.y_neuron_age.size())).repeat(lower_index[0].size()[0], 1)
            y_response = y_response.to(self.device)
            y_response.index_put_(
                indices=tuple(torch.stack([torch.arange(0, lower_index[1].size()[0]).to(self.device),
                                           lower_index[1]], dim=0)),
                values=(torch.ones(lower_index[1].size())).to(self.device))

            y_lr = torch.zeros(self.y_neuron_age.size())
            y_lr = y_lr.to(self.device)

            y_response.index_put_(
                indices=tuple(torch.stack([torch.arange(0, lower_index[1].size()[0]).to(self.device),
                                           lower_index[1]], dim=0)),
                values=max_response[lower_index[0]])

            idxs, cnts = torch.unique(lower_index[1], return_counts=True)
            divisor = torch.ones(self.y_neuron_age.size()[1])
            divisor = divisor.to(self.device)
            divisor.index_put_(indices=tuple(idxs.unsqueeze(0)), values=cnts.to(torch.float32))

            y_response = ((torch.sum(y_response, dim=0)).div(divisor)).unsqueeze(0)
            y_lr[0, idxs] = (1 / (self.y_neuron_age[0, idxs] + 1.0))

            self.y_threshold.data = y_lr.mul(y_response) + (1 - y_lr).mul(self.y_threshold.data)

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
        # self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=0)
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

    # def attention(self, max_index):
    #     att_weight = self.x2y.weight[max_index]
    #     att_min, _ = torch.min(att_weight, dim=1)
    #     att_max, _ = torch.max(att_weight, dim=1)
    #     att_weight = (att_weight - att_min.unsqueeze(1)) / (att_max.unsqueeze(1) - att_min.unsqueeze(1))
    #
    #     return att_weight

    # def attention(self, y_pre_response, y_activated, z):
    #     y_values, y_indices = torch.sort(y_pre_response, dim=1, descending=True)
    #
    #     top_k_values = y_values[:, :3]
    #     top_k_indices = y_indices[:, :3]
    #     top_k_act = y_activated.squeeze(0)[top_k_indices.reshape(-1)].reshape(top_k_indices.size())
    #     top_k_y2z_v, top_k_y2z_k = torch.max(self.y2z.weight.t()[top_k_indices.reshape(-1)], dim=1)
    #     judge_cls = torch.where(top_k_y2z_k.reshape(top_k_indices.size()) == z.unsqueeze(1).repeat(1, 3), 1, 0)
    #
    #     judge = judge_cls * top_k_act
    #
    #     values = top_k_values * top_k_y2z_v.reshape(top_k_indices.size()) * judge
    #     values = torch.where(values > 0, values, -10.0)
    #     values = F.softmax(values, dim=1).unsqueeze(2).repeat(1, 1, self.x2y.weight.data.size()[1])
    #
    #     att_weight = self.x2y.weight[top_k_indices.reshape(-1)].reshape((values.size()[0], values.size()[1], -1))
    #
    #     att_weight_output = torch.sum(values * att_weight, dim=1)
    #     # att_weight_output = F.softmax(att_weight_output, dim=1)
    #
    #     att_min, _ = torch.min(att_weight_output, dim=1)
    #     att_max, _ = torch.max(att_weight_output, dim=1)
    #     att_weight_output = (att_weight_output - att_min.unsqueeze(1)) / (att_max.unsqueeze(1) - att_min.unsqueeze(1))
    #
    #     return att_weight_output

    def attention(self, y_pre_response, y_activated, z, mode):
        top_k = 2

        y_values, y_indices = torch.sort(y_pre_response, dim=1, descending=True)

        top_k_values = y_values[:, :top_k]
        top_k_indices = y_indices[:, :top_k]
        top_k_act = y_activated.squeeze(0)[top_k_indices.reshape(-1)].reshape(top_k_indices.size())
        top_k_y2z_v, top_k_y2z_k = torch.max(self.y2z.weight.t()[top_k_indices.reshape(-1)], dim=1)

        if mode == 'train':
            judge_cls = torch.where(top_k_y2z_k.reshape(top_k_indices.size()) == z.unsqueeze(1).repeat(1, top_k), 1, 0)
            judge = judge_cls * top_k_act
        else:
            # judge = top_k_act
            judge_cls = torch.where(top_k_y2z_k.reshape(top_k_indices.size()) == z.unsqueeze(1).repeat(1, top_k), 1, 0)
            judge = judge_cls * top_k_act

        values = top_k_values * top_k_y2z_v.reshape(top_k_indices.size()) * judge
        values = torch.where(values > 0, values, -10.0)
        values = F.softmax(values, dim=1).unsqueeze(2).repeat(1, 1, self.x2y.weight.data.size()[1])

        att_weight = self.x2y.weight[top_k_indices.reshape(-1)].reshape((values.size()[0], values.size()[1], -1))

        att_weight_output = torch.sum(values * att_weight, dim=1)
        # att_weight_output = F.softmax(att_weight_output, dim=1)

        att_min, _ = torch.min(att_weight_output, dim=1)
        att_max, _ = torch.max(att_weight_output, dim=1)
        att_weight_output = (att_weight_output - att_min.unsqueeze(1)) / (att_max.unsqueeze(1) - att_min.unsqueeze(1))

        return att_weight_output
