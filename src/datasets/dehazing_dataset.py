import random
from torch.utils.data import Dataset
from PIL import Image

from src.config import Paths


class DehazingDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        clear_img_path = Paths.training_set / self.metadata[idx]['clear_image_path']
        hazy_img_paths = self.metadata[idx]['hazy_image_paths']

        # Load the clear image
        clear_img = Image.open(clear_img_path).convert('RGB')

        # Load the hazy image
        hazy_img = Image.open(
            Paths.training_set / random.choice(hazy_img_paths)
        ).convert('RGB')

        if self.transform is not None:
            clear_img = self.transform(clear_img)
            hazy_img = self.transform(hazy_img)

        # Return a tuple of the clear image and the hazy images
        return clear_img, hazy_img
