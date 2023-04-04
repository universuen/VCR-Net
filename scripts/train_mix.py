import context

import os
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.models import vgg19, VGG19_Weights
from torch import nn
from torchvision.utils import save_image

from src import api, config, DehazingDataset, models

api.set_seed(0)
logger = api.get_logger('train_script')


# MRF energy function
def mrf_energy(hazy_image, dehazed_image, alpha=0.1):
    hazy_grad_x = torch.abs(hazy_image[:, :, :-1, :] - hazy_image[:, :, 1:, :])
    hazy_grad_y = torch.abs(hazy_image[:, :, :, :-1] - hazy_image[:, :, :, 1:])

    dehazed_grad_x = torch.abs(dehazed_image[:, :, :-1, :] - dehazed_image[:, :, 1:, :])
    dehazed_grad_y = torch.abs(dehazed_image[:, :, :, :-1] - dehazed_image[:, :, :, 1:])

    energy = torch.sum(torch.abs(hazy_grad_x - dehazed_grad_x)) + torch.sum(torch.abs(hazy_grad_y - dehazed_grad_y))
    return alpha * energy


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


preprocess = transforms.Compose(
    [
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

tr_dataset = DehazingDataset(
    path=config.Paths.training_set,
    transform=preprocess,
)

# te_dataset = datasets.TestDataset(
#     path=config.Paths.test_set,
#     transform=preprocess,
# )

te_dataset = DehazingDataset(
    path=config.Paths.test_set,
    transform=preprocess,
)

vae = api.get_vae()
ae = models.Autoencoder().to(config.device)
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    *list(vgg.features.children())[1:17],
    nn.Flatten(),
).to(config.device)
feature_extractor.eval()

epsilon = nn.Parameter(torch.tensor(1.0))

vae_optimizer = optim.Adam(
    params=vae.parameters(),
    lr=config.Training.vae_learning_rate,
)
ae_optimizer = optim.Adam(
    params=ae.parameters(),
    lr=config.Training.ae_learning_rate,
)
epsilon_optimizer = optim.Adam(
    params=[epsilon],
    lr=config.Training.epsilon_learning_rate,
)

tr_dataloader = DataLoader(
    dataset=tr_dataset,
    batch_size=config.Training.batch_size,
    shuffle=True,
)

for e in range(config.Training.epochs):
    avg_loss = 0
    vae.train()
    ae.train()
    for idx, (clear_imgs, hazy_imgs) in enumerate(tr_dataloader):
        vae_optimizer.zero_grad()
        ae_optimizer.zero_grad()
        epsilon_optimizer.zero_grad()

        clear_imgs = clear_imgs.to(config.device)
        hazy_imgs = hazy_imgs.to(config.device)
        x_1, mu, log_var = vae(hazy_imgs)
        x_2 = ae(hazy_imgs)
        dehazed_imgs = (epsilon * x_1 + x_2) + hazy_imgs

        reconstruct_loss = nn.functional.mse_loss(dehazed_imgs, clear_imgs)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        r_clear = feature_extractor(clear_imgs)
        r_pred = feature_extractor(dehazed_imgs)
        r_hazy = feature_extractor(hazy_imgs)
        contrastive_loss = torch.mean(
            torch.cosine_similarity(r_pred, r_hazy) - torch.cosine_similarity(r_pred, r_clear),
            dim=0,
        )
        energy_mrf = mrf_energy(hazy_imgs, dehazed_imgs)
        loss = reconstruct_loss \
               + config.Training.beta * kl_divergence \
               + config.Training.alpha * contrastive_loss \
               + config.Training.gama * energy_mrf

        loss.backward()
        vae_optimizer.step()
        ae_optimizer.step()
        epsilon_optimizer.step()

        avg_loss += loss.item()
    avg_loss /= len(tr_dataloader)
    logger.info(f'epoch: {e + 1}: avg_loss: {avg_loss}, epsilon: {epsilon.item()}')

    # test
    if (e + 1) % 10 == 0:
        if not os.path.exists(config.Paths.test_results / f'e{e + 1}'):
            os.mkdir(config.Paths.test_results / f'e{e + 1}')
        with torch.no_grad():
            vae.eval()
            ae.eval()
            test_imgs = random.choices(te_dataset, k=3)
            for idx, (clear_img, hazy_img) in enumerate(test_imgs):
                hazy_img = torch.unsqueeze(hazy_img, 0).to(config.device)
                x_1, mu, log_var = vae(hazy_img)
                x_2 = ae(hazy_img)
                dehazed_img = (epsilon * x_1 + x_2) + hazy_img
                dehazed_img = torch.squeeze(dehazed_img)
                with open(config.Paths.test_results / f'e{e + 1}' / f'{idx}_hazy.png', 'wb') as f:
                    save_image(denormalize(hazy_img), f)
                with open(config.Paths.test_results / f'e{e + 1}' / f'{idx}_dehazed.png', 'wb') as f:
                    save_image(denormalize(dehazed_img), f)
                with open(config.Paths.test_results / f'e{e + 1}' / f'{idx}_clear.png', 'wb') as f:
                    save_image(denormalize(clear_img), f)
