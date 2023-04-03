import random
import os
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image


class DehazingDataset(Dataset):
    def __init__(self, path: Path, transform=None):
        self.path = path
        self.transform = transform
        self.hazy_imgs = os.listdir(path / 'clean')
        self.clear_imgs = os.listdir(path / 'clean')

    def __len__(self):
        return len(self.hazy_imgs)

    def __getitem__(self, idx):
        clear_img = Image.open(self.path / 'clean' / self.clear_imgs[idx]).convert('RGB')
        hazy_img = Image.open(self.path / 'hazy' / self.hazy_imgs[idx]).convert('RGB')

        if self.transform is not None:
            clear_img = self.transform(clear_img)
            hazy_img = self.transform(hazy_img)

        # Return a tuple of the clear image and the hazy images
        return clear_img, hazy_img
