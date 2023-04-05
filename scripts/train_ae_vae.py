import context

import os
import random
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.models import vgg19, VGG19_Weights
from torch import nn
from torchvision.utils import save_image

from src import api, config, DehazingDataset, models, metrics

logger = api.get_logger('train_script')


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

te_dataset = DehazingDataset(
    path=config.Paths.test_set,
    transform=preprocess,
)

ae = models.Autoencoder().to(config.device)
vae = api.get_vae()
epsilon = nn.Parameter(torch.tensor(1.0))
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    *list(vgg.features.children())[1:17],
    nn.Flatten(),
).to(config.device)
feature_extractor.eval()

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
te_dataloader = DataLoader(
    dataset=te_dataset,
    batch_size=1,
    shuffle=False,
)

for e in range(config.Training.epochs):
    avg_recon_loss = 0
    avg_contra_loss = 0
    for idx, (clear_imgs, hazy_imgs) in enumerate(tr_dataloader):
        vae_optimizer.zero_grad()
        ae_optimizer.zero_grad()
        epsilon_optimizer.zero_grad()

        clear_imgs = clear_imgs.to(config.device)
        hazy_imgs = hazy_imgs.to(config.device)
        x_1, mu, log_var = vae(hazy_imgs)
        x_2 = ae(hazy_imgs)
        x = epsilon * x_1 + x_2
        dehazed_imgs = x + hazy_imgs
        reconstruct_loss = nn.functional.mse_loss(dehazed_imgs, clear_imgs)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        r_clear = feature_extractor(clear_imgs)
        r_pred = feature_extractor(dehazed_imgs)
        r_hazy = feature_extractor(hazy_imgs)
        contrastive_loss = torch.mean(
            torch.cosine_similarity(r_pred, r_hazy) - torch.cosine_similarity(r_pred, r_clear),
            dim=0,
        )

        loss = reconstruct_loss + config.Training.alpha * contrastive_loss + config.Training.beta * kl_divergence
        loss.backward()
        vae_optimizer.step()
        ae_optimizer.step()
        epsilon_optimizer.step()

        avg_recon_loss += reconstruct_loss
        avg_contra_loss += contrastive_loss

    avg_recon_loss /= len(tr_dataloader)
    avg_contra_loss /= len(tr_dataloader)
    logger.info(
        f'epoch: {e + 1}: '
        f'avg_recon_loss: {avg_recon_loss:.4f}, '
        f'avg_contra_loss: {avg_contra_loss:.4f}, '
    )

    ### Evaluate and Generate the vis results
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        if (e + 1) % 10 == 0:
            ae.eval()
            vae.eval()
            # epsilon.eval()
            for idx, (clear_imgs, hazy_imgs) in enumerate(te_dataloader):
                clear_imgs = clear_imgs.to(config.device)
                hazy_imgs = hazy_imgs.to(config.device)
                x_1, mu, log_var = vae(hazy_imgs)
                x_2 = ae(hazy_imgs)
                x = epsilon * x_1 + x_2
                dehazed_imgs = x + hazy_imgs              
                # Gen eval results
                psnr = metrics.psnr(dehazed_imgs, clear_imgs)
                ssim = metrics.ssim(dehazed_imgs, clear_imgs).item()
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                # import ipdb; ipdb.set_trace()
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)

            # print(f'Average PSNR: {avg_psnr:.4f}\t', f'Average SSIM: {avg_ssim:.4f}')

            logger.info(
                f'epoch: {e + 1}: '
                f'Average PSNR: {avg_psnr:.4f}, '
                f'Average SSIM: {avg_ssim:.4f}, '
    )

        ### Random select image to generate visilization results    
        if (e + 1) % 30 == 0:
            if not os.path.exists(config.Paths.test_results / f'e{e + 1}'):
                os.mkdir(config.Paths.test_results / f'e{e + 1}')
            test_imgs = random.choices(te_dataset, k=3)
            for idx, (clear_imgs, hazy_imgs) in enumerate(test_imgs):
                img = hazy_imgs
                dehazed_img = torch.squeeze(
                    epsilon* vae(torch.unsqueeze(img, 0).to(config.device))[0] + \
                    ae(torch.unsqueeze(img, 0).to(config.device))[0] +img.to(config.device)
                ).detach().cpu()
                with open(config.Paths.test_results / f'e{e + 1}' / f'{idx}_hazy.png', 'wb') as f:
                    save_image(denormalize(img), f)
                with open(config.Paths.test_results / f'e{e + 1}' / f'{idx}_dehazed.png', 'wb') as f:
                    save_image(denormalize(dehazed_img), f)
                with open(config.Paths.test_results / f'e{e + 1}' / f'{idx}_clear.png', 'wb') as f:
                    save_image(denormalize(clear_imgs), f)
            ae.train()
            vae.train()
            # epsilon.train()
