'''
@File    :   train_cifar10_v2_1.py
@Time    :   2023/04/25 10:00
@Author  :   Hong Shijie
@Version :   2.1
@related :   cnn_dn_v1 dn_torch_v6...
@Note    :   在v2的基础上，使其适应于bath的训练
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

def get_dir(saver_dir):
    cnt = 0
    while True:
        saver_f = os.path.join(saver_dir, 'cifar10_v2d1_')
        if os.path.exists(saver_f + str(cnt)):
            cnt += 1
        else:
            break
    saver = saver_f + str(cnt)
    os.mkdir(saver)
    return saver

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

# model.load_state_dict(torch.load('../results/cifar10_v2_0/model_e41_acc37.pth'))

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
optimizer_b = torch.optim.SGD(model.backbone.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=10, gamma=0.5)

optimizer_d = torch.optim.SGD(model.dn.parameters(), lr=learning_rate)

save_dir = get_dir(save_path)

for epo in range(epochs):
    loop_train = tqdm(train_loader, ncols=120)
    for img, label in loop_train:
        img, label = img.to(model.device), label.to(model.device)
        img, label = Variable(img), Variable(label, requires_grad=False)

        output, activated_num = model(img, label, 'train', dn_items)

        loss = criterion(output, label)
        optimizer_b.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_b.step()

        loop_train.set_description(f"Epoch: {epo}")
        loop_train.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0],
                               activated_number=activated_num.item(), memory='{}MB'.format(int(torch.cuda.max_memory_allocated() / 1024 / 1024)))

    loop_test = tqdm(test_loader, ncols=80)
    num_correct = 0
    total_loss = 0
    for img, label in loop_test:
        img, label = img.to(model.device), label.to(model.device)
        with torch.no_grad():
            output = model(img, label, 'test', dn_items)
            num_correct += (output.argmax(1) == label).sum()

            acc = float(num_correct) / len(test_loader.dataset)
            loop_test.set_description(f"Test:")
            loop_test.set_postfix(acc=acc, memory='{}MB'.format(int(torch.cuda.memory_allocated() / 1024 / 1024)))

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_e{}_acc{}.pth'.format(epo, int(acc * 100))))
    scheduler.step()


