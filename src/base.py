from dataclasses import dataclass
import logging
from pprint import pprint
from traceback import print_exc
from typing import Any, Dict, List
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from turtle import forward

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset


@dataclass
class BaseConfig:
    learning_rate: float = 1e-4
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_epochs: int = 500
    loss: str = "mse"
    device: str = "cuda"


def switch_dict_tensor_device(d: Dict, device: str):
    for k, v in d.items():
        if torch.is_tensor(v) and v.device != device:
            d[k] = v.to(device)


class BaseTrainer:
    def __init__(self, config: BaseConfig) -> None:
        self.config = config

    def get_optimizer(self):
        pass

    def setup(self, config):
        pass

    def teardown(self):
        pass

    def train(self):
        self.step = 0
        self.setup(self.config)

        try:
            for e in range(self.config.num_epochs):
                self.run_train_epoch(e)
                self.evaluate(e)
        except KeyboardInterrupt:
            logging.info("Interrupted!")
        except Exception as e:
            print_exc()

        self.teardown()

    def run_train_epoch(self, epoch: int):
        optim = self.get_optimizer()
        device = self.config.device

        train_loader = DataLoader(
            self.train_dataset, self.config.eval_batch_size, shuffle=True
        )

        desc = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in desc:
            switch_dict_tensor_device(batch, device)

            loss = self.train_step(batch)
            loss.backward()

            desc.desc = f"epoch {epoch}, loss={loss.item()}"

            optim.step()
            optim.zero_grad()

            self.step += 1

    def train_step(self, batch: Dict) -> Dict:
        """
        loss 는 무조건 있어야한다.
        """
        pass

    @torch.no_grad()
    def evaluate(self, epoch: int):
        pass
