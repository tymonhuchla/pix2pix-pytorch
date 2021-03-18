from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import zipfile
import os
from config import *


with zipfile.ZipFile(PATH, 'r') as zip_ref:
    if 'data' not in os.listdir():
        os.mkdir('data')
    zip_ref.extractall('./data')


class ConcatDatasets(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return min(len(d) for d in self.datasets)

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)


transform_a = transforms.Compose([
                                  transforms.Resize((SHAPE, SHAPE)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5], std=[0.5])
                                  ])

data_a = ImageFolder(rootdir, transform=transform_a)

transform_b = transforms.Compose([
                                  transforms.Resize((SHAPE, SHAPE)),
                                  transforms.Grayscale(num_output_channels=1),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5], std=[0.5])
                                  ])

data_b = ImageFolder(rootdir, transform_b)

data_full = ConcatDatasets(data_b, data_a)

loader = DataLoader(data_full, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)