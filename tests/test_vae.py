import context

import torch

from src import config, api


vae = api.get_vae()

images = torch.randn(64, 3, 64, 64, device=config.device)

output = vae(images)

print(output)
