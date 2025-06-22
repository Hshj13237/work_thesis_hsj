'''
@File    :   train_cifar10_v2_3.py
@Time    :   2023/05/23 16:00
@Author  :   Hong Shijie
@Version :   2.3
@related :   cnn_dn_v1 dn_torch_v6_7
@Note    :   在v2.1的基础上,对该轮中已经激活的神经元进行移除判定，当该神经元在新的一轮学习中从未被激活，则将其年龄与阈值,z权重清除；
'''
import os

import torch.utils.data
import torchvision
from tqdm import tqdm
from torch import nn
from utils.concat.cnn_dn_v1 import concat_net
from torch.autograd import Variable
import pickle

epochs = 20
batch_size = 32
dn_input_dim = [1, 512]
y_neuron_num = 50000
y_top_k = 1
y_bottom_up_percent = 0.5
z_neuron_num = 10
synapse_flag = False
dn_items = 1
learning_rate = 0.005
save_path = '../results'


def main():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../data/cifar10', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                         # torchvision.transforms.Grayscale(num_output_channels=1),
                                         torchvision.transforms.ToTensor(),
                                         # torchvision.transforms.Normalize(
                                         #     (0,), (1,))
                                         # (0.1307,), (0.3081,))
                                     ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../data/cifar10', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                         # torchvision.transforms.Grayscale(num_output_channels=1),
                                         torchvision.transforms.ToTensor(),
                                         # torchvision.transforms.Normalize(
                                         #     (0,), (1,))
                                         # (0.1307,), (0.3081,))
                                     ])),
        batch_size=batch_size, shuffle=True)

    model = concat_net(batch_size, dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag)
    model = model.to(model.device)

    state_load = torch.load('../results/model_e60_acc92.pth')
    state_load.pop('resnet18.fc.weight')
    state_load.pop('resnet18.fc.bias')
    model.backbone.load_state_dict(state_load)

    optimizer_b = torch.optim.SGD(model.backbone.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=10, gamma=0.5)

    optimizer_d = torch.optim.SGD(model.dn.parameters(), lr=learning_rate)

    save_dir = get_dir(save_path)

    for epo in range(epochs):

        # "多组循环训练"
        # if epo % 2 == 0:
        #     temp_y_neuron_age = model_dn.dn.y_neuron_age.data.cpu()
        #     temp_z_neuron_age = model_dn.dn.z_neuron_age.data.cpu()
        #     temp_y_threshold = model_dn.dn.y_threshold.data.cpu()
        # else:
        #     model_dn.dn.y_neuron_age.data = temp_y_neuron_age.cuda()
        #     model_dn.dn.z_neuron_age.data = temp_z_neuron_age.cuda()
        #     model_dn.dn.y_threshold.data = temp_y_threshold.cuda()

        train(train_loader, model, optimizer_b, scheduler, optimizer_d, epo)
        remove(model, epo, save_dir)

        acc = test(train_loader, model, 'train')
        acc = test(test_loader, model, 'valid')

        torch.save(model.state_dict(), os.path.join(save_dir, 'model_e{}_acc{}.pth'.format(epo, int(acc * 100))))
        scheduler.step()


def get_dir(saver_dir):
    cnt = 0
    while True:
        saver_f = os.path.join(saver_dir, 'cifar10_v2d3_')
        if os.path.exists(saver_f + str(cnt)):
            cnt += 1
        else:
            break
    saver = saver_f + str(cnt)
    os.mkdir(saver)
    return saver


def train(train_loader, model, optimizer_b, scheduler, optimizer_d, epo):
    loop_train = tqdm(train_loader, ncols=150)

    num_correct = 0
    criterion = nn.CrossEntropyLoss()

    for img, label in loop_train:
        img, label = img.to(model.device), label.to(model.device)
        img, label = Variable(img), Variable(label, requires_grad=False)

        output, activated_num = model(img, label, 'train', dn_items)

        num_correct += (output.argmax(1) == label).sum()
        acc = float(num_correct) / len(train_loader.dataset)

        loss = criterion(output, label)
        optimizer_b.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_b.step()

        loop_train.set_description(f"Epoch: {epo}")
        loop_train.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0], activated_number=activated_num.item(),
                               acc=acc, memory='{}MB'.format(int(torch.cuda.max_memory_allocated() / 1024 / 1024)))


def remove(model_dn, epo, save_dir):
    if epo > 0:
        last_y_neuron_age = torch.load(os.path.join(save_dir, 'y_neuron_age_e{}.pth'.format(epo-1)))
        # last_z_neuron_age = torch.load(os.path.join(save_dir, 'z_neuron_age_e{}.pth'.format(epo-1))).cuda()
        # last_y_threshold = torch.load(os.path.join(save_dir, 'y_threshold_e{}.pth'.format(epo-1))).cuda()

        judge1 = (model_dn.dn.y_neuron_age.data.cpu() - last_y_neuron_age).squeeze(0)
        judge1 = torch.where(judge1 == 0, 1, 0)

        judge2 = torch.ones(model_dn.dn.y_neuron_age.data.size()).squeeze(0)
        judge2.index_put_(indices=tuple(model_dn.dn.max_index.unsqueeze(0).to(torch.int64)), values=torch.zeros(model_dn.dn.max_index.size()))

        judge3 = torch.where(model_dn.dn.y_neuron_age.data.cpu() == 0, 0, 1).squeeze(0)

        temp_y_indices = torch.nonzero(judge1 * judge2 * judge3 == 1).t()

        y_activated = torch.where(model_dn.dn.y_neuron_age >= 1, 1, 0).cpu()
        y_activated = int(torch.sum(y_activated))
        model_dn.dn.z_neuron_age.data = torch.floor(
            (1 - len(temp_y_indices.squeeze(0)) / y_activated) * model_dn.dn.z_neuron_age.data)

        model_dn.dn.y_neuron_age.data.index_put_(
            indices=tuple(torch.cat((torch.zeros(temp_y_indices.size()), temp_y_indices), dim=0).to(torch.int64).cuda()),
            values=torch.zeros(temp_y_indices.squeeze(0).size()).cuda())

        model_dn.dn.y_threshold.data.index_put_(
            indices=tuple(torch.cat((torch.zeros(temp_y_indices.size()), temp_y_indices), dim=0).to(torch.int64).cuda()),
            values=torch.zeros(temp_y_indices.squeeze(0).size()).cuda())

        model_dn.dn.y2z.weight.data[:, temp_y_indices.squeeze(0).numpy()] = 0

    torch.save(model_dn.dn.y_neuron_age.data.cpu(), os.path.join(save_dir, 'y_neuron_age_e{}.pth'.format(epo)))
    model_dn.dn.max_index = torch.empty(0)


def test(test_loader, model, name):
    loop_test = tqdm(test_loader, ncols=120)
    num_correct = 0
    total_loss = 0

    for img, label in loop_test:
        img, label = img.to(model.device), label.to(model.device)
        with torch.no_grad():
            output = model(img, label, 'test', dn_items)
            num_correct += (output.argmax(1) == label).sum()

            acc = float(num_correct) / len(test_loader.dataset)
            loop_test.set_description("Test({}):".format(name))
            loop_test.set_postfix(acc=acc, memory='{}MB'.format(int(torch.cuda.memory_allocated() / 1024 / 1024)))

    return acc


if __name__ == '__main__':
    main()

