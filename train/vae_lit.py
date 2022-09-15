import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

from .base import BaseLitModule




class VAELitModule(BaseLitModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

