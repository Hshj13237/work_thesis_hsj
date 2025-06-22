import subprocess

file_list = ['train_cifar100_v1_1_1.py', 'train_cifar100_v1_1D1.py']

for file in file_list:
    subprocess.call(['python', file])
