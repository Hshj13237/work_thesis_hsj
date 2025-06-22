'''
@File    :   dn_torch_v3.py
@Time    :   2024/04/19 16:00
@Author  :   Hong Shijie
@Version :
@Note    :   高层指导低层
'''

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import random


class Trans_y_response(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pre, y_act, y_act_mis):
        # y_act_neg = y_act * 2 - 1.0
        ctx.save_for_backward(y_act_mis)
        return y_act

    @staticmethod
    def backward(ctx, grad_output):
        y_act, = ctx.saved_tensors
        grad_y_pre = grad_output.mul(y_act)
        # grad_y_pre = torch.zeros(grad_output.size()).cuda()
        grad_y_act = None
        grad_y_act_mis = None
        return grad_y_pre, grad_y_act, grad_y_act_mis


class DN(nn.Module):
    def __init__(self, input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag, input_dim_2):
        super(DN, self).__init__()
        self.x_neuron_num = input_dim[0] * input_dim[1]
        self.x_neuron_num_2 = input_dim_2[0] * input_dim_2[1]
        self.y_neuron_num = y_neuron_num
        self.z_neuron_num = z_neuron_num + 1
        self.y_top_k = y_top_k
        self.y_bottom_up_percent = y_bottom_up_percent
        self.y_top_down_percent = 1.0 - self.y_bottom_up_percent
        self.synapse_age = 1.0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.x2y = nn.Linear(self.x_neuron_num, self.y_neuron_num, bias=False)

        self.z2y = nn.Linear(self.z_neuron_num, self.y_neuron_num, bias=False)

        self.y2z = nn.Linear(self.y_neuron_num, self.z_neuron_num, bias=False)
        self.x2y2 = nn.Linear(32768, self.y_neuron_num, bias=False)
        self.x2y3 = nn.Linear(320, self.y_neuron_num, bias=False)
        self.x2y4 = nn.Linear(512, self.y_neuron_num, bias=False)


        self.y_neuron_age = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)

        self.y_threshold = nn.Parameter(torch.zeros((1, self.y_neuron_num)), requires_grad=False)

        self.z_neuron_age = nn.Parameter(torch.zeros((1, self.z_neuron_num)), requires_grad=False)

        self.trans_y = Trans_y_response.apply

        self.x2y.weight.data = torch.rand(self.x2y.weight.data.shape)
        self.z2y.weight.data = torch.rand(self.z2y.weight.data.shape)

        # self.y2z.weight.data = torch.rand(self.y2z.weight.data.shape)
        self.y2z.weight.data = torch.zeros(self.y2z.weight.data.shape)

        self.x2y2.weight.data = torch.rand(self.x2y2.weight.data.shape)
        self.x2y3.weight.data = torch.rand(self.x2y3.weight.data.shape)
        self.x2y4.weight.data = torch.rand(self.x2y4.weight.data.shape)

        self.z2y.weight.requires_grad = False


        self.max_index = torch.empty(0)
        # self.max_res = torch.empty(0)
        # self.cos_sim = torch.empty(0)
        self.sim_thr = 0.95

        self.alpha = 0.94

        self.gray_trans = transforms.Grayscale(1)

        self.mini_batch = 4

        self.temp_save = []

    def forward(self, x, z, mode, per_item, epo, x2, x3, x4):
        z = z + 1
        batch = int(x.shape[0])
        img = x.data
        x = x.view(batch, -1)
        x = F.normalize(x, dim=1)
        z_hot = F.one_hot(z, num_classes=self.z_neuron_num)

        z_hot = F.normalize(z_hot.float(), dim=1)

        # if self.y_neuron_num >= 2000:
        #     self.mini_batch = 64
        # elif self.y_neuron_num >= 3000:
        #     self.mini_batch = 32
        # elif self.y_neuron_num >= 4000:
        #     self.mini_batch = 16
        # elif self.y_neuron_num >= 15000:
        #     self.mini_batch = 8
        # elif self.y_neuron_num >= 17000:
        #     self.mini_batch = 4

        # x2 = x2.view(batch, -1)
        # x2 = F.normalize(x2, dim=1)
        # x3 = x3.view(batch, -1)
        # x3 = F.normalize(x3, dim=1)
        # x4 = x4.view(batch, -1)
        # x4 = F.normalize(x4, dim=1)

        if mode == 'lock_backbone':
            with torch.no_grad():
                for num in range(per_item):

                    self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
                    self.z2y.weight.data = F.normalize(self.z2y.weight.data, dim=1)
                    self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)
                    # self.x2y2.weight.data = F.normalize(self.x2y2.weight.data, dim=1)
                    # self.x2y3.weight.data = F.normalize(self.x2y3.weight.data, dim=1)
                    # self.x2y4.weight.data = F.normalize(self.x2y4.weight.data, dim=1)

                    y_bottom_up_response = self.x2y(x)

                    # y_bottom_up_response_2 = self.x2y4(x4)
                    y_top_down_response = self.z2y(z_hot)

                    # y_pre_response = self.y_bottom_up_percent * y_bottom_up_response + \
                    #                  (0.5 - self.y_bottom_up_percent) * y_bottom_up_response_2 + \
                    #                  self.y_top_down_percent * y_top_down_response
                    # y_pre_response = 0.35 * y_bottom_up_response + 0.15 * y_bottom_up_response_2 + 0.5 * y_top_down_response
                    y_pre_response = 0.5 * y_bottom_up_response + 0.5 * y_top_down_response

                    max_response, max_index, lower_index = self.top_k_competition(y_pre_response, mode, z)
                    self.hebbian_learning(x, z_hot, max_index, max_response, batch, per_item, lower_index, num, x2, x3, x4)

                y_activated = torch.where(self.y_neuron_age >= 1, 1, 0)
                y_activated = y_activated.to(self.device)

                # self.sim_del(y_pre_response, y_activated, epo)
                # self.max_res = torch.cat((self.max_res, max_response.data.cpu()), dim=0)

                y_activated_num = torch.sum(y_activated)

                return y_activated_num

        elif mode == 'lock_dn':
            self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
            self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)
            x2y = self.x2y(x)
            y_activated = torch.where(self.y_neuron_age >= 1, 1, 0)
            y_activated = y_activated.to(self.device)
            y_temp = x2y.mul(y_activated)
            y_temp_cp = y_temp.data
            y = self.top_k_competition(y_temp_cp, mode, z)

            var = []
            var.append(self.x2y2.weight.data[torch.argmax(y, dim=1), :])
            var.append(self.x2y3.weight.data[torch.argmax(y, dim=1), :])

            var_loss = F.mse_loss(self.x2y2.weight.data[torch.argmax(y, dim=1), :], x2) +  \
                       F.mse_loss(self.x2y3.weight.data[torch.argmax(y, dim=1), :], x3)
                       # F.mse_loss(self.x2y4.weight.data[torch.argmax(y, dim=1), :], x4)

            y_mis = self.cal_grad(y, z_hot)

            y_tr = self.trans_y(y_temp, y, y_mis)

            output = self.y2z(y_tr)

            return output, var_loss, var

        elif mode == 'test':
            self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
            # self.x2y2.weight.data = F.normalize(self.x2y2.weight.data, dim=1)
            # self.x2y3.weight.data = F.normalize(self.x2y3.weight.data, dim=1)
            self.x2y4.weight.data = F.normalize(self.x2y4.weight.data, dim=1)
            # self.y2z.weight.data = F.normalize(self.y2z.weight.data, dim=1)

            # y_bottom_up_response = self.x2y4(x4)
            y_bottom_up_response = self.x2y(x)
            # y_bottom_up_response = self.y_bottom_up_percent * 2 * self.x2y(x) + (1 - 2 * self.y_bottom_up_percent) * self.x2y2(img)
            # y_bottom_up_response = 0.1 * self.x2y(x) + 0.9 * self.x2y2(x2)
            y_activated_num = torch.where(self.y_neuron_age >= 1, 1, 0)
            y_activated_num = y_activated_num.to(self.device)
            y_pre_response = y_bottom_up_response.mul(y_activated_num)
            y_response = self.top_k_competition(y_pre_response, mode, z)
            output = self.y2z(y_response)
            # output = torch.argmax(output[0])
            return output

    def top_k_competition(self, y_pre_response, mode, z):
        if mode == 'lock_backbone':
            max_response, max_index = torch.max(y_pre_response, 1)

            y_activated = torch.where(self.y_neuron_age >= 1, 1, -1)

            # if self.y2z.weight[:,torch.max(y_pre_response * y_activated, 1)] !=

            "存储最大激活值的索引"
            temp_mask = ~torch.isin(max_index.data.cpu(), self.max_index)
            temp_indices, _ = torch.sort(torch.unique(max_index.data.cpu()[temp_mask]))
            self.max_index = torch.cat((self.max_index, temp_indices), dim=0)

            'test'
            # self.y_neuron_age[0, [max_index[5], max_index[9]]] = 1.0

            # cor_judge = torch.argmax(self.top_k_competition(y_pre_response, 'test', z), dim=1)
            # cor_judge = torch.where(cor_judge == z, 1, 0)

            judge = torch.where(max_response >= (self.y_threshold[0, max_index.data] * self.alpha), 1, 0) \
                | torch.where(self.y_neuron_age[0, max_index.data] < self.synapse_age, 1, 0)

            add_index = torch.where(judge < 1)
            # add_index = torch.stack(add_index)
            lower_index = torch.cat((torch.stack(add_index), max_index[add_index].unsqueeze(0)), dim=0)

            if torch.sum(judge) < len(judge):
                unactivated_flag = torch.where(self.y_neuron_age >= 1, 0., 1.)

                if torch.sum(unactivated_flag, dim=1) > 0:
                    y_temp_response = y_pre_response[torch.stack(add_index), :].squeeze(0)
                    y_temp_response = torch.mul(y_temp_response, unactivated_flag)
                    max_index.index_put_(indices=add_index, values=torch.argmax(y_temp_response, dim=1))
                    # max_index.index_copy_(0, add_index.squeeze(0), torch.argmax(y_temp_response, dim=1))

            # if torch.sum(judge) < len(judge):
            #     y_temp_response = y_pre_response[torch.stack(add_index), :].squeeze(0)
            #     useful_flag = torch.where(y_temp_response >= self.y_threshold.data.repeat(y_temp_response.size()[0], 1),
            #                              1, 0)
            #     y_temp_response = torch.mul(y_temp_response, useful_flag)
            #     max_index.index_put_(indices=add_index, values=torch.argmax(y_temp_response, dim=1))

            return max_response, max_index, lower_index

        elif mode == 'lock_dn':
            y_response = torch.zeros(y_pre_response.shape)
            y_response = y_response.to(self.device)

            max_response, max_index = torch.max(y_pre_response, 1)

            index = tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0))
            y_response.index_put_(indices=index, values=(torch.ones(max_index.size())).to(self.device))

            return y_response

        elif mode == 'test':
            y_response = torch.zeros(y_pre_response.shape)
            y_response = y_response.to(self.device)

            y_activated_num = torch.where(self.y_neuron_age >= 1, 1, 0)
            y_pre_response = torch.mul(y_pre_response, y_activated_num)

            y_data, y_indices = torch.sort(y_pre_response, dim=1, descending=True)

            max_index = y_indices[:, 0]

            #最值表
            _ , max_y2z = torch.max(self.y2z.weight.data, dim=0)

            ch_idx = torch.nonzero(max_y2z[max_index] == 0).squeeze(1)

            un_idx = torch.nonzero(y_data[torch.arange(0, max_index.size()[0]).to(self.device), 0] == 0.0).squeeze(1)
            mask = ~torch.isin(ch_idx, un_idx)
            ch_idx = ch_idx[mask]
            i = 0

            while ch_idx.numel():
                i = i + 1
                judge = torch.where(y_data[ch_idx, i] != 0.0, True, False) & torch.where(
                    max_y2z[y_indices[ch_idx, i]] != 0, True, False)

                if torch.sum(judge) > 0:
                    max_index[ch_idx[judge]] = y_indices[ch_idx[judge], i]

                un_idx = ch_idx[torch.nonzero(y_data[ch_idx, i] == 0.0).squeeze(1)]
                ch_idx = ch_idx[torch.nonzero(max_y2z[y_indices[ch_idx, i]] == 0).squeeze(1)]
                mask = ~torch.isin(ch_idx, un_idx)
                ch_idx = ch_idx[mask]

            judge = torch.where(y_data[:, 0] != 0.0, True, False) & torch.where(max_y2z[max_index] != z, True, False)
            ch_idx = torch.nonzero(judge == True)
            for x in range(2):
                judge = torch.where(y_data[ch_idx, x+1] != 0.0, True, False) & torch.where(
                    max_y2z[y_indices[ch_idx, x+1]] == z[ch_idx], True, False)
                if torch.sum(judge) > 0:
                    cur_idx = ch_idx[judge]
                    judge = torch.where(y_data[cur_idx, 0] - y_data[cur_idx, 1] < 0.01, True, False)
                    if torch.sum(judge) > 0:
                        self.temp_save.append((max_y2z[max_index[cur_idx[judge]]].data.cpu().numpy(),
                                               max_y2z[y_indices[cur_idx[judge], x + 1]].data.cpu().numpy()))
                        max_index[cur_idx[judge]] = y_indices[cur_idx[judge], x+1]
                    mask = ~torch.isin(ch_idx, cur_idx)
                    ch_idx = ch_idx[mask]

            # y2z_data, _ = torch.max(self.y2z.weight[:, max_index].data, dim=0)
            # ch_idx = torch.nonzero(y2z_data.data < 0.3).squeeze(1)
            #
            # for i in ch_idx:
            #     j = 1
            #     y2z_data, _ = torch.max(self.y2z.weight[:, y_indices[i, j]].data, dim=0)
            #     while y2z_data < 0.1:
            #         j += 1
            #         y2z_data, _ = torch.max(self.y2z.weight[:, y_indices[i, j]].data, dim=0)
            #     max_index[i] = y_indices[i, j]

            # ch_idx = torch.nonzero(torch.sum(self.y2z.weight[:, max_index].data == 0.0, dim=0) <
            #                         (self.z_neuron_num - 1)).squeeze(1)
            #
            # for i in ch_idx:
            #     j = 1
            #     while torch.sum(self.y2z.weight[:, y_indices[i, j]].data == 0.0, dim=0) != (self.z_neuron_num - 1):
            #         j += 1
            #         if y_pre_response[i, j] == 0:
            #             j = -1
            #             break
            #     if j == -1:
            #         continue
            #     else:
            #         max_index[i] = y_indices[i, j]

            index = tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0))
            y_response.index_put_(indices=index, values=(torch.ones(max_index.size())).to(self.device))

            return y_response

    def hebbian_learning(self, x, z_hot, max_index, max_response, batch, per_item, lower_index, num, x2, x3, x4):
        y_response = (torch.zeros(self.y_neuron_age.size())).repeat(batch, 1)
        y_response = y_response.to(self.device)

        # y_response.index_put_(
        #     indices=tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0)),
        #     values=(torch.ones(max_index.size())).to(self.device))

        y_response.index_put_(
            indices=tuple(torch.stack([torch.arange(0, max_index.size()[0]).to(self.device), max_index], dim=0)),
            values=max_response)


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

        # for i in range(int(batch / self.mini_batch)):
        #     temp_weight = ((1 - y_lr[i * self.mini_batch:(i+1) * self.mini_batch].repeat(1, 1, self.x2y.weight.data.shape[1])).mul(self.x2y.weight.data.unsqueeze(0)) +
        #                    y_lr[i * self.mini_batch:(i+1) * self.mini_batch].repeat(1, 1, self.x2y.weight.data.shape[1]).mul(x[i * self.mini_batch:(i+1) * self.mini_batch].unsqueeze(1))) - \
        #         self.x2y.weight.data.unsqueeze(0).repeat(self.mini_batch, 1, 1)
        #     temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))
        #
        #     self.x2y.weight.data = self.x2y.weight.data + temp_weight

        # x2 to y
        for i in range(int(batch / self.mini_batch * 4)):
            temp_weight = ((1 - y_lr[i * int(self.mini_batch/4):(i+1) * int(self.mini_batch / 4)].repeat(1, 1, self.x2y2.weight.data.shape[1])).mul(self.x2y2.weight.data.unsqueeze(0)) +
                           y_lr[i * int(self.mini_batch/4):(i+1) * int(self.mini_batch / 4)].repeat(1, 1, self.x2y2.weight.data.shape[1]).mul(x2[i * int(self.mini_batch/4):(i+1) * int(self.mini_batch / 4)].unsqueeze(1))) - \
                self.x2y2.weight.data.unsqueeze(0).repeat(int(self.mini_batch/4), 1, 1)
            temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))

            self.x2y2.weight.data = self.x2y2.weight.data + temp_weight

        # x3 to y
        for i in range(int(batch / self.mini_batch * 4)):
            temp_weight = ((1 - y_lr[i * int(self.mini_batch/4):(i+1) * int(self.mini_batch / 4)].repeat(1, 1, self.x2y3.weight.data.shape[1])).mul(self.x2y3.weight.data.unsqueeze(0)) +
                           y_lr[i * int(self.mini_batch/4):(i+1) * int(self.mini_batch / 4)].repeat(1, 1, self.x2y3.weight.data.shape[1]).mul(x3[i * int(self.mini_batch/4):(i+1) * int(self.mini_batch / 4)].unsqueeze(1))) - \
                self.x2y3.weight.data.unsqueeze(0).repeat(int(self.mini_batch/4), 1, 1)
            temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))

            self.x2y3.weight.data = self.x2y3.weight.data + temp_weight

        # x4 to y
        for i in range(int(batch / self.mini_batch * 4)):
            temp_weight = ((1 - y_lr[i * int(self.mini_batch/4):(i+1) * int(self.mini_batch / 4)].repeat(1, 1, self.x2y4.weight.data.shape[1])).mul(self.x2y4.weight.data.unsqueeze(0)) +
                           y_lr[i * int(self.mini_batch/4):(i+1) * int(self.mini_batch / 4)].repeat(1, 1, self.x2y4.weight.data.shape[1]).mul(x4[i * int(self.mini_batch/4):(i+1) * int(self.mini_batch / 4)].unsqueeze(1))) - \
                self.x2y4.weight.data.unsqueeze(0).repeat(int(self.mini_batch/4), 1, 1)
            temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))

            self.x2y4.weight.data = self.x2y4.weight.data + temp_weight

        # z to y
        temp_weight = ((1 - y_lr.repeat(1, 1, self.z2y.weight.data.shape[1])).mul(self.z2y.weight.data.unsqueeze(0)) +
                       y_lr.repeat(1, 1, self.z2y.weight.data.shape[1]).mul(z_hot.unsqueeze(1))) - \
            self.z2y.weight.data.unsqueeze(0).repeat(batch, 1, 1)
        temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))

        self.z2y.weight.data = self.z2y.weight.data + temp_weight


        # for i in range(int(batch / self.mini_batch)):
        #     temp_weight = ((1 - y_lr[i * self.mini_batch:(i+1) * self.mini_batch].repeat(1, 1, self.z2y.weight.data.shape[1])).mul(self.z2y.weight.data.unsqueeze(0)) +
        #                    y_lr[i * self.mini_batch:(i+1) * self.mini_batch].repeat(1, 1, self.z2y.weight.data.shape[1]).mul(z_hot[i * self.mini_batch:(i+1) * self.mini_batch].unsqueeze(1))) - \
        #         self.z2y.weight.data.unsqueeze(0).repeat(self.mini_batch, 1, 1)
        #     temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))
        #
        #     self.z2y.weight.data = self.z2y.weight.data + temp_weight

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

            self.y_neuron_age.data = torch.where(self.y_neuron_age.data > 20, 20, self.y_neuron_age.data)

            # y to z
            # z_lr = (1 / (self.z_neuron_age + 1.0))
            # z_lr = torch.where(self.z_neuron_age > 10, 0.1, 1 / (self.z_neuron_age + 1.0))

            max_v, max_id = torch.max(self.y2z.weight.data, dim=0)
            eff_id = torch.nonzero(max_v)
            all_id = torch.cat([max_id[eff_id[:, 0]], torch.arange(0, self.z_neuron_num).to(self.device)])
            _, y2z_counts = torch.unique(all_id, return_counts=True)
            z_lr = (1 / y2z_counts.unsqueeze(0))

            # z_lr = 0.5 - self.z_neuron_age / (torch.sum(self.z_neuron_age) + 1)
            # z_lr = (1 / (self.z_neuron_age + 1.0))
            z_lr = z_lr.to(self.device)
            z_lr = z_hot.mul(z_lr).unsqueeze(2)

            cnts = torch.sum(z_hot, dim=0)
            divisor = torch.where(cnts > 0, cnts, 1.0)

            temp_weight = ((1 - z_lr.repeat(1, 1, self.y2z.weight.data.shape[1])).mul(self.y2z.weight.data.unsqueeze(0)) +
                           z_lr.repeat(1, 1, self.y2z.weight.data.shape[1]).mul(y_response.unsqueeze(1))) - \
                self.y2z.weight.data.unsqueeze(0).repeat(batch, 1, 1)

            # y_lr = (1 / (self.y_neuron_age + 1.0)).unsqueeze(1).repeat(batch, 1, 1)
            # z_lr = y_lr * z_hot.unsqueeze(2).repeat(1, 1, self.y2z.weight.data.shape[1])
            # z_lr = z_lr.to(self.device)
            # cnts = torch.sum(z_hot, dim=0)
            # divisor = torch.where(cnts > 0, cnts, 1.0)
            #
            # temp_weight = ((1 - z_lr).mul(self.y2z.weight.data.unsqueeze(0)) + z_lr.mul(y_response.unsqueeze(1))) - \
            #     self.y2z.weight.data.unsqueeze(0).repeat(batch, 1, 1)


            temp_weight = (torch.sum(temp_weight, dim=0)).div(divisor.unsqueeze(1))
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

                # self.y_threshold.data = torch.where(self.y_threshold.data > 0.9, 0.9, self.y_threshold.data)

    def reset(self, new_num):
        self.y_neuron_num = new_num

        temp_data = self.x2y.weight.data
        self.x2y = nn.Linear(self.x_neuron_num, self.y_neuron_num, bias=False)
        self.x2y.weight.data = torch.rand(self.x2y.weight.data.shape)
        # self.x2y.weight.data = F.normalize(self.x2y.weight.data, dim=1)
        # self.x2y.weight.data[temp_data.size()[0]:-1, :] = temp_data
        # self.x2y.weight.data[temp_data.size()[0]:-1, :] = temp_data
        self.x2y.weight[:temp_data.size()[0]].data.copy_(temp_data)

        temp_data = self.x2y2.weight.data
        self.x2y2 = nn.Linear(self.x2y2.in_features, self.y_neuron_num, bias=False)
        self.x2y2.weight.data = torch.rand(self.x2y2.weight.data.shape)
        self.x2y2.weight[:temp_data.size()[0]].data.copy_(temp_data)

        temp_data = self.x2y3.weight.data
        self.x2y3 = nn.Linear(self.x2y3.in_features, self.y_neuron_num, bias=False)
        self.x2y3.weight.data = torch.rand(self.x2y3.weight.data.shape)
        self.x2y3.weight[:temp_data.size()[0]].data.copy_(temp_data)

        temp_data = self.x2y4.weight.data
        self.x2y4 = nn.Linear(self.x2y4.in_features, self.y_neuron_num, bias=False)
        self.x2y4.weight.data = torch.rand(self.x2y4.weight.data.shape)
        self.x2y4.weight[:temp_data.size()[0]].data.copy_(temp_data)

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

    def sim_del(self, y_pre_response, y_activated, epo):
        top_k = 2

        # self.sim_thr = 0.6 + (epo % 50) * 0.05
        # self.sim_thr = 0.7 + 0.05 * epo
        #
        # if self.sim_thr >= 0.95:
        #     self.sim_thr = 0.95

        _, y_indices = torch.sort(y_pre_response, dim=1, descending=True)
        top_k_indices = y_indices[:, :top_k]
        top_k_act = y_activated.squeeze(0)[top_k_indices.reshape(-1)].reshape(top_k_indices.size())
        top_k_x2y = self.x2y.weight[top_k_indices.reshape(-1)].reshape(
            top_k_indices.size()[0], top_k_indices.size()[1], -1)
        cos_sim = F.cosine_similarity(top_k_x2y[:, 0, :], top_k_x2y[:, 1, :], dim=1)

        judge = torch.nonzero(cos_sim > self.sim_thr).squeeze(1)
        del_idx = top_k_indices[judge, 1].unsqueeze(0).cpu()

        y_activated = int(torch.sum(y_activated.cpu()))

        self.z_neuron_age.data = torch.floor(
            (1 - len(del_idx.squeeze(0)) / y_activated) * self.z_neuron_age.data)

        self.y_neuron_age.data.index_put_(
            indices=tuple(
                torch.cat((torch.zeros(del_idx.size()), del_idx), dim=0).to(torch.int64).cuda()),
            values=torch.zeros(del_idx.squeeze(0).size()).cuda())

        self.y_threshold.data.index_put_(
            indices=tuple(
                torch.cat((torch.zeros(del_idx.size()), del_idx), dim=0).to(torch.int64).cuda()),
            values=(torch.zeros(del_idx.squeeze(0).size())).cuda())

        self.y2z.weight.data[:, del_idx.squeeze(0).numpy()] = 0
        self.x2y.weight.data[del_idx.squeeze(0).numpy(), :] = torch.rand((
            len(del_idx.squeeze(0)), self.x2y.weight.size()[1])).cuda()

        # self.z2y.weight.data[del_idx.squeeze(0).numpy(), :] = torch.rand((
        #     len(del_idx.squeeze(0)), self.z2y.weight.size()[1])).cuda()

        # self.cos_sim = torch.cat((self.cos_sim, cos_sim.cpu()), dim=0)

    def cal_grad(self, y, z):
        with torch.no_grad():
            out = self.y2z(y)
            judge = (out.argmax(1) != z.argmax(1)).unsqueeze(1)
            y = y * judge
            return y

from matplotlib import pyplot as plt
def att_vis(x, h, w, i):
    fig = plt.figure(figsize=(40, 15))
    # ax = fig.add_subplot(2, 14, i + 2, xticks=[], yticks=[])
    x = x.view(128, h, w)
    x_v = (x[i] - (0.7*x[i].max() + 0.3*x[i].min())).clamp(0).data.cpu().numpy()
    plt.imshow(x_v)
    plt.show()


