import random
import csv
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image


class DehazingDataset(Dataset):
    def __init__(self, path: Path, transform=None):
        self.path = path
        self.metadata = []
        with open(path / 'metadata.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                hazy_image_paths = row['hazy_image_paths'].strip('[]').split(', ')
                hazy_image_paths = [path.strip("'") for path in hazy_image_paths]
                self.metadata.append(
                    {
                        'image_id': int(row['image_id']),
                        'clear_image_path': row['clear_image_path'],
                        'hazy_image_paths': hazy_image_paths
                    }
                )
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        clear_img_path = self.path / self.metadata[idx]['clear_image_path']
        hazy_img_paths = self.metadata[idx]['hazy_image_paths']

        # Load the clear image
        clear_img = Image.open(clear_img_path).convert('RGB')

        # Load the hazy image
        hazy_img = Image.open(
            self.path / random.choice(hazy_img_paths)
        ).convert('RGB')

        if self.transform is not None:
            clear_img = self.transform(clear_img)
            hazy_img = self.transform(hazy_img)

        # Return a tuple of the clear image and the hazy images
        return clear_img, hazy_img