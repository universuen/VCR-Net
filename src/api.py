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
