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
from collections import defaultdict
import wandb


def switch_dict_tensor_device(d: dict, device: str):
    for k, v in d.items():
        if torch.is_tensor(v) and v.device != device:
            d[k] = v.to(device)

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

def pad_sentence_latents(sents: torch.Tensor, max_seq_len: int):
    """
        sents: [s, dim]

        out: [max_s, dim]
    """
    pad_size = max_seq_len - sents.shape[0]
    if pad_size == 0:
        return sents
    elif pad_size < 0:
        return sents[:max_seq_len]
    else:
        pad = torch.zeros(
            (pad_size, sents.shape[1]),
            device = sents.device,
            dtype=sents.dtype
        )
        return torch.cat([sents, pad], dim=0)


class SGPTTask(BaseTask):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        gpt2_config = GPT2Config.from_json_file(to_absolute_path("config/model/" + config.model.model))
        self.autoencoder = OptimusTask.load_from_checkpoint(to_absolute_path(config.model.autoencoder)).eval()
        freeze_model(self.autoencoder)
        self.model = GPT2Model(gpt2_config)

    
    def get_train_collator(self) -> Callable:
        return lambda x: x

    def get_eval_collator(self) -> Callable:
        return lambda x: x

    def get_train_dataset(self) -> Dataset:
        return SGPTDataset(
            # self.autoencoder,
            self.config.dataset.train,
            self.config.model.max_seq_len,
            column="utterances",
            weights=self.config.dataset.get("train_weights")
        )

    def get_eval_dataset(self) -> Dataset:
        return SGPTDataset(
            # self.autoencoder,
            self.config.dataset.test,
            self.config.model.max_seq_len,
            column="utterances",
            split="test"
        )

    def _item_for_train(self, sents):
        sents = self.autoencoder.encode(sents)
        mask_len = min(len(sents), self.config.model.max_seq_len)
        attention_mask = [1] * mask_len + [0] * (self.config.model.max_seq_len - mask_len)

        sents = pad_sentence_latents(sents, self.config.model.max_seq_len + 1)
        
        return {
            "inputs": sents[:-1],
            "attention_mask": torch.LongTensor(attention_mask),
            "labels": sents[1:]
        }

    def step(self, batch, batch_idx) -> dict:
        new_batch = defaultdict(list)
        for b in batch:
            item = self._item_for_train(b["utterances"])
            for k, v in item.items():
                new_batch[k].append(v)
        
        batch = {}
        for k, v in new_batch.items():
            batch[k] = torch.stack(v).to(self.device)

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
        self.log_dict(out, prefix="val_", on_epoch=True, batch_size=len(batch))

        # if batch_idx < 2:
        #     print(batch["sentences"])
        #     seq_lens = batch["attention_mask"].cpu().sum(-1)
        #     for i, seq_len in enumerate(seq_lens):
        #         recons = self.autoencoder.generate(outputs[i, :seq_len])
        #         for src, rec in zip(batch["sentences"][i], recons):
        #             print(batch_idx, i, src, rec)