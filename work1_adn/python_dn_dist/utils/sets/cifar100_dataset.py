from torch.utils.data import Dataset
from PIL import Image

class Cifar100Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        target = self.targets[item]
        img = Image.fromarray(self.data[item])
        if self.transform:
            img = self.transform(img)
        return img, target

# aquatic:
# fish:
#flowers:
#food:
#fruit and vegetables:
# household
