from typing import Any
from pydantic import BaseModel
from omegaconf import DictConfig

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import pytorch_lightning as pl



class BaseTask(BaseModel, pl.LightningModule):
    config: DictConfig

    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)
        __pydantic_self__.setup(__pydantic_self__.config)

    def setup(self, config: DictConfig):
        pass

    def get_optimizer(self):
        oc = self.config.optimizer

        if oc.cls == "adam":
            return optim.Adam(self.parameters(), lr=oc.learning_rate)
        else:
            raise Exception(f"{oc.cls} is unsupported optimizer type.")
    
    def get_scheduler(self):
        return None

    def configure_optimizers(self):
        return self.get_optimizer(), self.get_scheduler()


    def get_train_dataset(self) -> Dataset:
        pass

    def get_eval_dataset(self) -> Dataset:
        return None

    # def log_dict(self, *args, **kwargs):
    #     prefix = kwargs.get("prefix", None)
        
    #     if prefix is not None:
    #         kwargs["dictionary"] = {prefix + k:v for k, v in kwargs["dictionary"].items()}

    #     super().log_dict(*args, **kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.get_train_dataset(),
            batch_size=self.config.trainer.train_batch_size,
            shuffle=self.config.trainer.get("shuffle", True),
            num_workers=self.config.trainer.get("num_workers", 4)
        )

    def val_dataloader(self):
        dataset = self.get_eval_dataset()
        if dataset is not None:
            return DataLoader(
                dataset,
                batch_size=self.config.trainer.eval_batch_size,
                num_workers=self.config.trainer.get("num_workers", 4)
            )
        else:
            return None

    @classmethod
    def main(cls, config: DictConfig):
        ckpt = config.get("checkpoint", None)

        if ckpt is None:
            task = cls(config)
        else:
            task = cls.load_from_checkpoint(ckpt)

        trainer = pl.Trainer(
            max_epochs=config.trainer.train_epochs
        )
        trainer.fit(task)
