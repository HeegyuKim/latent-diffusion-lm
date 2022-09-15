from typing import Optional

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from coop.models import Optimus


def get_optimizer(config: DictConfig, params):
    oc = config.optimizer

    if oc.cls == "adam":
        return optim.Adam(params, lr=oc.learning_rate)
    else:
        raise Exception(f"{oc.cls} is unsupported optimizer type.")


def get_scheduler(config: DictConfig):
    return None


def get_model(config: DictConfig):
    mc = config.model

    if mc.cls == "optimus":
        return Optimus(
            latent_dim=mc.latent_dim,
            bos_id=50256,
            pad_id=50256,
            eos_id=50256
        )
    else:
        raise Exception(f"{mc.cls} is unsupported model type.")

def get_dataset_split(config: Optional[DictConfig]):
    if config is None:
        return None

    return load_dataset(**config)

def get_dataset(config: DictConfig):
    dc = config.dataset
    out = {}
    out["train"] = get_dataset_split(dc.train)
    out["eval"] = get_dataset_split(dc.get("eval"))

    return out

def get_trainer(config: DictConfig):
    return pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        logger=pl.loggers.WandbLogger(),
        log_every_n_steps=1,
    )

class BaseLitModule(pl.LightningModule):
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = get_model(config)

    def configure_optimizers(self):
        optim = get_optimizer(self.config, self.parameters())
        sched = get_scheduler(self.config)
        return optim, sched

