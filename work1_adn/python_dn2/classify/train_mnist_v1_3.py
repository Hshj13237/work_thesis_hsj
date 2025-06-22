'''
@File    :   train_mnist_v1_3.py
@Time    :   2023/04/25 22:00
@Author  :   Hong Shijie
@Version :   1.3
@Note    :   在v1.1的基础上,使其可自主循环不同batch
'''

import torch.utils.data
import torchvision
import utils.dn.dn_torch_v5 as DN
from tqdm import tqdm

epochs = 10
batch_size_group = [32, 16, 8, 4, 1]
input_dim = [28, 28]
y_neuron_num = 45000
y_top_k = 1
y_bottom_up_percent = 0.5
z_neuron_num = 10
synapse_flag = False

for batch_size in batch_size_group:
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data/mnist', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       # torchvision.transforms.Normalize(
                                       #     (0,), (1,))
                                           # (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data/mnist', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0,), (1,))
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
                activated_num = dn(img, label, 'train', 1)
                loop_train.set_description(f"Epoch: {epo}")
                loop_train.set_postfix(activated_number=activated_num.item(), memory='{}MB'.format(int(torch.cuda.max_memory_allocated() / 1024 / 1024)))

            loop_test = tqdm(test_loader, ncols=100)
            num_correct = 0
            cnt = 0
            for img, label in loop_test:
                # torch.cuda.empty_cache()
                img = img.to(dn.device)
                label = label.to(dn.device)
                output = dn(img, label, 'test', 1)
                num_correct += (output.argmax(1) == label).sum()
                # num_correct += (output == label).sum()

                # acc = float(num_correct) / float(len(test_loader))
                # acc = float(num_correct) / float(cnt)
                acc = float(num_correct) / len(test_loader.dataset)
                loop_test.set_description(f"Test:")
                loop_test.set_postfix(cor=int(num_correct.cpu().numpy()), acc=acc, memory='{}MB'.format(int(torch.cuda.memory_allocated() / 1024 / 1024)))

    torch.cuda.empty_cache()


