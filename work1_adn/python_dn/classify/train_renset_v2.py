'''
@File    :   train_resnet.py
@Time    :   2023/03/21 20:00
@Author  :   Hong Shijie
@Version :   1.0
@Note    :   resnet18 for cifar10
'''
import torch.utils.data
import torchvision
from tqdm import tqdm
from torch import nn
from utils.cnn.backbone_v3 import ResNet18
import os

epochs = 500
batch_size = 128
input_dim = [32, 32]
y_neuron_num = 55000
y_top_k = 1
y_bottom_up_percent = 0.5
z_neuron_num = 100
synapse_flag = False
learning_rate = 0.1
save_path = '../results'

def get_dir(saver_dir):
    cnt = 0
    while True:
        saver_f = os.path.join(saver_dir, 'resnet18_v3_')
        if os.path.exists(saver_f + str(cnt)):
            cnt += 1
        else:
            break
    saver = saver_f + str(cnt)
    os.mkdir(saver)
    return saver

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('../data/cifar100', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   # torchvision.transforms.Grayscale(num_output_channels=1),
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize(
                                   #     (0,), (1,))
                                       # (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('../data/cifar100', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   # torchvision.transforms.Grayscale(num_output_channels=1),
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize(
                                   #     (0,), (1,))
                                       # (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

# train_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageNet('../data/ImageNet_1k', split='train', download=True,
#                                transform=torchvision.transforms.Compose([
#                                    # torchvision.transforms.Grayscale(num_output_channels=1),
#                                    torchvision.transforms.ToTensor(),
#                                    # torchvision.transforms.Normalize(
#                                    #     (0,), (1,))
#                                        # (0.1307,), (0.3081,))
#                                ])),
#     batch_size=batch_size, shuffle=False)
#
# test_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageNet('../data/ImageNet_1k', split='val', download=True,
#                                transform=torchvision.transforms.Compose([
#                                    # torchvision.transforms.Grayscale(num_output_channels=1),
#                                    torchvision.transforms.ToTensor(),
#                                    # torchvision.transforms.Normalize(
#                                    #     (0,), (1,))
#                                        # (0.1307,), (0.3081,))
#                                ])),
#     batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

backbone = ResNet18(z_neuron_num)
backbone = backbone.to(device)

state_load = backbone.load_pretrained_weights()
backbone.load_state_dict(state_load, strict=False)

# state_load = torch.load('../results/model_e246_acc726.pth')
# backbone.load_state_dict(state_load)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(backbone.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

save_dir = get_dir(save_path)

for epo in range(epochs):
    backbone.train()
    loop_train = tqdm(train_loader, ncols=100)
    for img, label in loop_train:
        img, label = img.to(device), label.to(device)

        output = backbone(img)

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop_train.set_description(f"Epoch: {epo}")
        loop_train.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0], memory='{}MB'.format(int(torch.cuda.memory_allocated() / 1024 / 1024)))

    scheduler.step()

    backbone.eval()

    loop_test = tqdm(train_loader, ncols=150)
    num_correct = 0
    total_loss = 0
    with torch.no_grad():
        for img, label in loop_test:
            img, label = img.to(device), label.to(device)

            output = backbone(img)
            loss = criterion(output, label)
            total_loss += loss.item()
            num_correct += (output.argmax(1) == label).sum()

            acc = float(num_correct) / len(train_loader.dataset)
            epoch_loss = float(total_loss) / len(train_loader.dataset)
            loop_test.set_description(f"\tTest(train):")
            loop_test.set_postfix(acc=acc, loss=epoch_loss, memory='{}MB'.format(int(torch.cuda.memory_allocated() / 1024 / 1024)))

    loop_test = tqdm(test_loader, ncols=150)
    num_correct = 0
    total_loss = 0
    with torch.no_grad():
        for img, label in loop_test:
            img, label = img.to(device), label.to(device)

            output = backbone(img)
            loss = criterion(output, label)
            total_loss += loss.item()
            num_correct += (output.argmax(1) == label).sum()

            acc = float(num_correct) / len(test_loader.dataset)
            epoch_loss = float(total_loss) / len(test_loader.dataset)
            loop_test.set_description(f"\tTest(Valid):")
            loop_test.set_postfix(acc=acc, loss=epoch_loss, memory='{}MB'.format(int(torch.cuda.memory_allocated() / 1024 / 1024)))

        torch.save(backbone.state_dict(), os.path.join(save_dir, 'model_e{}_acc{}.pth'.format(epo, int(acc * 100))))
