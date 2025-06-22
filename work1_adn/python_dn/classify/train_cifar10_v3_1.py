'''
@File    :   train_cifar10_v3_2.py
@Time    :   2023/05/23 12:30
@Author  :   Hong Shijie
@Version :   3.1
@related :   cnn_dn_v1 dn_torch_v6_7
@Note    :   在v3的基础上,对该轮中已经激活的神经元进行移除判定，当该神经元在新的一轮学习中从未被激活，则将其年龄与阈值,z权重清除
'''
import os

import torch.utils.data
import torchvision
from tqdm import tqdm
from torch import nn
from utils.concat.cnn_dn_v1 import concat_net
from  utils.cnn.backbone_v1 import backbone
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
learning_rate = 0.001
save_path = '../results'
global_step = 0


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

    model_dn = concat_net(batch_size, dn_input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num,
                          synapse_flag)
    # model_dn = nn.DataParallel(model_dn).cuda()
    model_dn = model_dn.cuda()

    model_fc = backbone(z_neuron_num, batch_size)
    # model_fc = nn.DataParallel(model_fc).cuda()
    model_fc = model_fc.cuda()

    state_load = torch.load('../results/model_e60_acc92.pth')
    model_fc.load_state_dict(state_load)
    state_load.pop('resnet18.fc.weight')
    state_load.pop('resnet18.fc.bias')
    model_dn.backbone.load_state_dict(state_load)

    optimizer = torch.optim.SGD(model_fc.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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

        train(train_loader, model_dn, model_fc, optimizer, scheduler, epo)
        remove(model_dn, epo, save_dir)

        test(train_loader, model_dn, model_fc, 'train')
        fc_acc, dn_acc = test(test_loader, model_dn, model_fc, 'valid')

        torch.save(model_fc.state_dict(), os.path.join(save_dir, 'model_fc_e{}_acc{}.pth'.format(epo, int(fc_acc * 100))))
        torch.save(model_dn.state_dict(), os.path.join(save_dir, 'model_dn_e{}_acc{}.pth'.format(epo, int(dn_acc * 100))))
        scheduler.step()


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

    torch.save(model_dn.dn.y_neuron_age.data.cpu(), os.path.join(save_dir, 'y_neuron_age_e{}.pth'.format(epo)))
    model_dn.dn.max_index = torch.empty(0)


def get_dir(saver_dir):
    cnt = 0
    while True:
        saver_f = os.path.join(saver_dir, 'cifar10_v3d1_')
        if os.path.exists(saver_f + str(cnt)):
            cnt += 1
        else:
            break
    saver = saver_f + str(cnt)
    os.mkdir(saver)
    return saver


def train(train_loader, model_dn, model_fc, optimizer, scheduler, epo):
    global global_step

    loop_train = tqdm(train_loader, ncols=120)

    class_criterion = nn.CrossEntropyLoss()
    consistency_criterion = nn.MSELoss()

    for img, label in loop_train:
        img, label = img.to(model_dn.device), label.to(model_dn.device)
        img, label = Variable(img), Variable(label, requires_grad=False)

        output_dn, activated_num = model_dn(img, label, 'train', dn_items)
        output_fc = model_fc(img)

        class_loss_fc = class_criterion(output_fc, label)
        class_loss_dn = class_criterion(output_dn, label)
        consistency_loss = consistency_criterion(output_dn, output_fc)
        loss = class_loss_fc + class_loss_dn + consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model_fc, model_dn, 0.999, global_step)

        loop_train.set_description(f"Epoch: {epo}")
        loop_train.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0],
                               activated_number=activated_num.item(), memory='{}MB'.format(int(torch.cuda.max_memory_allocated() / 1024 / 1024)))


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)

    for ema_param, param in zip(ema_model.backbone.parameters(), model.parameters()):
    # for (name1, ema_param), (name2, param) in zip(ema_model.backbone.named_parameters(), model.named_parameters()):
    #     print(name1, '\t', name2)
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def test(test_loader, model_dn, model_fc, name):
    loop_test = tqdm(test_loader, ncols=120)
    fc_num_correct = 0
    dn_num_correct = 0

    for img, label in loop_test:
        img, label = img.to(model_dn.device), label.to(model_dn.device)
        with torch.no_grad():
            fc_output = model_fc(img)
            dn_output = model_dn(img, label, 'test', dn_items)

            fc_num_correct += (fc_output.argmax(1) == label).sum()
            dn_num_correct += (dn_output.argmax(1) == label).sum()

            fc_acc = float(fc_num_correct) / len(test_loader.dataset)
            dn_acc = float(dn_num_correct) / len(test_loader.dataset)

            loop_test.set_description("Test({}):".format(name))
            loop_test.set_postfix(fc_acc=fc_acc, dn_acc=dn_acc, memory='{}MB'.format(int(torch.cuda.memory_allocated() / 1024 / 1024)))
    return fc_acc, dn_acc


if __name__ == '__main__':
    main()
