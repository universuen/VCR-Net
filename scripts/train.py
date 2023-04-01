import context

import csv

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.models import vgg19, VGG19_Weights
from torch import nn

from src import api, config, datasets

logger = api.get_logger('train_script')

metadata = []

with open(config.Paths.training_set / 'metadata.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        # Extract the hazy image paths and convert them into a list
        hazy_image_paths = row['hazy_image_paths'].strip('[]').split(', ')
        hazy_image_paths = [path.strip("'") for path in hazy_image_paths]

        # Add the current example to the metadata list
        metadata.append({
            'image_id': int(row['image_id']),
            'clear_image_path': row['clear_image_path'],
            'hazy_image_paths': hazy_image_paths
        })

preprocess = transforms.Compose(
    [
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

tr_dataset = datasets.DehazingDataset(
    metadata=metadata,
    transform=preprocess,
)

vae = api.get_vae()
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    *list(vgg.features.children())[1:17],
    nn.Flatten(),
).to(config.device)
feature_extractor.eval()

optimizer = optim.Adam(
    params=vae.parameters(),
    lr=config.Training.learning_rate,
)
tr_dataloader = DataLoader(
    dataset=tr_dataset,
    batch_size=config.Training.batch_size,
    shuffle=True,
)

for e in range(config.Training.epochs):
    avg_loss = 0
    for idx, (clear_imgs, hazy_imgs) in enumerate(tr_dataloader):
        clear_imgs = clear_imgs.to(config.device)
        hazy_imgs = hazy_imgs.to(config.device)
        optimizer.zero_grad()
        x, mu, log_var = vae(hazy_imgs)
        reconstruct_loss = nn.functional.mse_loss(x, clear_imgs)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        r_clear = feature_extractor(clear_imgs)
        r_pred = feature_extractor(x)
        r_hazy = feature_extractor(hazy_imgs)
        contrastive_loss = torch.mean(
            torch.cosine_similarity(r_pred, r_hazy) - torch.cosine_similarity(r_pred, r_clear),
            dim=0,
        )
        loss = reconstruct_loss + config.Training.beta * kl_divergence + config.Training.alpha * contrastive_loss
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    avg_loss /= len(tr_dataloader)
    logger.info(f'epoch: {e + 1}: avg_loss: {avg_loss}')
