import random

import torch
import numpy as np

import src
from src import config, types


def get_logger(name: str) -> types.Logger:
    return src.logger.Logger(
        name=name,
        level=src.config.Logger.level,
        logs_dir=src.config.Paths.logs,
    )


def get_vae() -> types.VAE:
    return src.models.VAE(1024).to(config.device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
