import subprocess

file_list = ['train_cifar100_v1.py', 'train_cifar100_v2.py']

for file in file_list:
    subprocess.call(['python', file])