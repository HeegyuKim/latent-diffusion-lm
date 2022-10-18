from typing import Any, Callable, List, Optional, Union
from .base import BaseTask
from omegaconf import DictConfig
from coop.models import Optimus
import pandas as pd
import torch
import torch.optim as optim
from torch import kl_div, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import GPT2Model, GPT2Config
from tokenizers import Tokenizer
from datasets import load_dataset

from hydra.utils import get_original_cwd, to_absolute_path

from ..dataset.sgpt import SGPTDataset
from .optimus_v2 import OptimusTask
from tqdm import tqdm
import wandb


def switch_dict_tensor_device(d: dict, device: str):
    for k, v in d.items():
        if torch.is_tensor(v) and v.device != device:
            d[k] = v.to(device)

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

class SGPTTask(BaseTask):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        gpt2_config = GPT2Config.from_json_file(to_absolute_path("config/model/" + config.model.model))
        self.autoencoder = OptimusTask.load_from_checkpoint(to_absolute_path(config.model.autoencoder)).eval()
        freeze_model(self.autoencoder)
        self.model = GPT2Model(gpt2_config)

    
    def get_train_dataset(self) -> Dataset:
        return SGPTDataset(
            self.autoencoder,
            self.config.dataset.train,
            self.config.model.max_seq_len,
            column="utterances",
            weights=self.config.dataset.get("train_weights")
        )

    def get_eval_dataset(self) -> Dataset:
        return SGPTDataset(
            self.autoencoder,
            self.config.dataset.test,
            self.config.model.max_seq_len,
            column="utterances",
            split="test"
        )
        
    def step(self, batch, batch_idx) -> dict:
        mask = batch["attention_mask"]
        hidden = self.model(
            inputs_embeds=batch["inputs"],
            attention_mask=mask
            ).last_hidden_state
        outputs = F.mse_loss(hidden, batch["labels"], reduce=False).mean(-1)
        outputs = outputs.masked_fill(mask == 0, 0)
        loss = outputs.sum() / mask.sum()

        return loss, hidden

    def training_step(self, batch, batch_idx) -> dict:
        loss, _ = self.step(batch, batch_idx)

        out = {"loss": loss}
        self.log_dict(out, prefix="train_")
        
        return out

    def validation_step(self, batch, batch_idx) -> dict:
        loss, outputs = self.step(batch, batch_idx)

        out = {"loss": loss}
        self.log_dict(out, prefix="val_", on_epoch=True)

        # if batch_idx < 2:
        #     print(batch["sentences"])
        #     seq_lens = batch["attention_mask"].cpu().sum(-1)
        #     for i, seq_len in enumerate(seq_lens):
        #         recons = self.autoencoder.generate(outputs[i, :seq_len])
        #         for src, rec in zip(batch["sentences"][i], recons):
        #             print(batch_idx, i, src, rec)