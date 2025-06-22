'''
@File    :   train_cifar10_v1.py
@Time    :   2023/03/19 23:00
@Author  :   Hong Shijie
@Version :   1.0
@Note    :   dn for cifar10
'''
import torch.utils.data
import torchvision
import utils.dn.dn_torch_v2 as DN
from tqdm import tqdm

epochs = 10
batch_size = 1
input_dim = [32, 32]
y_neuron_num = 50000
y_top_k = 1
y_bottom_up_percent = 0.5
z_neuron_num = 10
synapse_flag = False

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('../data/cifar10', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Grayscale(num_output_channels=1),
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize(
                                   #     (0,), (1,))
                                       # (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('../data/cifar10', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Grayscale(num_output_channels=1),
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize(
                                   #     (0,), (1,))
                                       # (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

dn = DN.DN(input_dim, y_neuron_num, y_top_k, y_bottom_up_percent, z_neuron_num, synapse_flag)
dn = dn.to(dn.device)

for epo in range(epochs):
    with torch.no_grad():
        loop_train = tqdm(train_loader, ncols=100)
        for img, label in loop_train:
            # torch.cuda.empty_cache()

            img = img.to(dn.device)
            label = label.to(dn.device)
            activated_num = dn(img, label, 'train', 2)
            loop_train.set_description(f"Epoch: {epo}")
            loop_train.set_postfix(activated_number=activated_num.item(), memory='{}MB'.format(int(torch.cuda.memory_allocated() / 1024 / 1024)))

        loop_test = tqdm(test_loader, ncols=80)
        num_correct = 0
        cnt = 0
        for img, label in loop_test:
            # torch.cuda.empty_cache()
            img = img.to(dn.device)
            label = label.to(dn.device)
            output = dn(img, label, 'test', 1)
            num_correct += (output.item() == label)
            cnt += 1
            # acc = float(num_correct) / float(len(test_loader))
            acc = float(num_correct) / float(cnt)
            loop_test.set_description(f"Test:")
            loop_test.set_postfix(acc=acc, memory='{}MB'.format(int(torch.cuda.memory_allocated() / 1024 / 1024)))
