import os
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image


class TestDataset(Dataset):
    def __init__(self, path: Path, transform=None):
        self.transform = transform
        self.imgs = [
            Image.open(path / i).convert('RGB')
            for i in os.listdir(path)
        ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(self.imgs[idx]) if self.transform is not None else self.imgs[idx]
