import torch
from omegaconf import DictConfig

from .giga_world_policy import GigaWorldPolicy


def get_model(cfg: DictConfig, torch_dtype=None):
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    return GigaWorldPolicy(cfg, torch_dtype=torch_dtype)
