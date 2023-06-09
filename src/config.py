from __future__ import annotations

import torch

from typing import Type
import logging
from pathlib import Path

"""
Attention: To guarantee the code's low-coupling and reusability, 
please do not import this module in any `src` modules except `api`.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class _Config:
    @classmethod
    def to_dict(cls) -> dict:
        result = dict()
        for k, v in vars(cls).items():
            if k.startswith('_') or k in ('i', 'j', 'k'):
                continue
            result[k] = v
        return result


ConfigType = Type[_Config]


class Paths(_Config):
    src: Path = Path(__file__).absolute().parent
    project: Path = src.parent
    data: Path = project / 'data'
    scripts: Path = project / 'scripts'
    tests: Path = project / 'tests'
    logs: Path = data / 'logs'
    training_set: Path = data / 'train'
    test_set: Path = data / 'test'
    test_results: Path = data / 'test_results'
    # create path if not exists
    for i in list(vars().values()):
        if isinstance(i, Path):
            i.mkdir(parents=True, exist_ok=True)


class Logger(_Config):
    level = logging.INFO


class VAE(_Config):
    latent_dim = 1024


class Training(_Config):
    alpha: float = 0.1
    beta: float = 0.01
    gamma: float = 0
    epochs: int = 1000
    batch_size: int = 2
    vae_learning_rate: float = 1e-4 #1e-3
    ae_learning_rate: float = 1e-4 #1e-3
    epsilon_learning_rate: float = 1e-4 #1e-3


_all_items = vars().values()


def get_all_configs() -> list[ConfigType]:
    results = []
    for i in _all_items:
        try:
            if issubclass(i, _Config) and not i.__name__.startswith('_'):
                results.append(i)
        except TypeError:
            pass
    return results
